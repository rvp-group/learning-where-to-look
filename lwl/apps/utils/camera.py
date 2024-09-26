import numpy as np
from lwl.apps.utils.general_utils import transform

class Camera(object):
    """
    pinhole camera projection model
    """
    def __init__(self, rows, cols, fx, fy, cx, cy, num_bins_per_dim=30):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.rows = rows
        self.cols = cols
        self.half_vfov = np.arctan2(self.rows/2.0, self.fy)
        self.half_hfov = np.arctan2(self.cols/2.0, self.fx)
        # binning
        self.num_bins_per_dim = num_bins_per_dim
        self.pixel_to_bin_img, self.max_bin_idx = self.precompute_binmap(num_bins_per_dim=num_bins_per_dim)
        self.binned_features = np.zeros((num_bins_per_dim*num_bins_per_dim), dtype=np.int32)

    def project3d(self, pt, min_depth=0.01):
        """
        Projects a 3D point onto a 2D image plane using the camera's intrinsic parameters.

        Args:
            pt (tuple or list): A 3D point represented as (x, y, z).
            min_depth (float, optional): The minimum depth value to consider for projection. 
                         Points with a z-value less than this will not be projected. 
                         Defaults to 0.01.

        Returns:
            numpy.ndarray or None: A 2D point represented as (u, v) if the projection is within the image bounds,
                       otherwise None.
        """
        x = pt[0]
        y = pt[1]
        z = pt[2]

        if pt[2] < min_depth:
            return None

        u = np.zeros(2)
        u[0] = (x / z * self.fx + self.cx) + 0.5
        u[1] = (y / z * self.fy + self.cy) + 0.5

        if u[0] < 0 or u[0] >= self.cols:
            return None
        if u[1] < 0 or u[1] >= self.rows:
            return None
        return u
    
    def precompute_binmap(self, num_bins_per_dim):
        """
        Precomputes a bin map for the camera image.

        This function divides the image into a grid of bins and assigns each pixel
        to a bin index. The number of bins along each dimension is specified by 
        `num_bins_per_dim`.

        Args:
            num_bins_per_dim (int): The number of bins along each dimension.

        Returns:
            tuple: A tuple containing:
            - pixel_to_bin_img (np.ndarray): A 2D array where each element is the bin index 
              corresponding to the pixel at that position.
            - max_bin_idx (int): The maximum bin index.
        """
        # set num bins along each dimensions
        bin_size_cols = self.cols // num_bins_per_dim 
        bin_size_rows = self.rows // num_bins_per_dim 
        pixel_to_bin_img = np.zeros((self.rows, self.cols), dtype=np.int32)
        for r in range(0, self.rows):
            r_bin = np.floor(r / bin_size_rows) * num_bins_per_dim
            for c in range(0, self.cols): 
                bin_idx = np.floor(r_bin + c / bin_size_cols)
                pixel_to_bin_img[r, c] = bin_idx  

        # find max bin idx
        max_r_bin = np.floor(self.rows / bin_size_rows) * num_bins_per_dim
        max_bin_idx = int(np.floor(max_r_bin + self.cols / bin_size_cols))
        return pixel_to_bin_img, max_bin_idx

    def checkVisibilityBatch(self, pcs, margin=0.01):
        """
        Check the visibility of a batch of 3D points.

        This method projects a batch of 3D points and checks their visibility 
        based on the given margin. It returns an array indicating the visibility 
        status of each point.

        Args:
            pcs (np.ndarray): A numpy array of shape (N, 3) representing the 
                              batch of 3D points to be checked.
            margin (float, optional): A margin value to be considered during 
                                      the projection. Default is 0.01.

        Returns:
            np.ndarray: An array of integers where each element is 1 if the 
                        corresponding 3D point is visible, and 0 otherwise.
        """
        assert pcs.shape[1] == 3
        res = self.project3dBatch(pcs, margin)

        return np.array([int(v is not None) for v in res])

    def checkVisibilityBatchWorld(self, Twc, pws, margin=0.01):
        """
        Checks the visibility of a batch of world points from the camera's perspective.

        Parameters:
        Twc (numpy.ndarray): A 4x4 transformation matrix representing the pose of the camera in the world frame.
        pws (numpy.ndarray): An array of 3D points in the world frame.
        margin (float, optional): A margin value for visibility check. Default is 0.01.

        Returns:
        numpy.ndarray: An array of integers where 1 indicates the point is visible and 0 indicates it is not.
        """
        T_c_w = np.linalg.inv(Twc)
        pcs = np.array([transform(T_c_w, v) for v in pws])
        assert pcs.shape[1] == 3
        res = self.project3dBatch(pcs, margin)

        return np.array([int(v is not None) for v in res])

    def project3DWorld(self, Twc, pw):
        """
        Projects a 3D world point into the camera frame.

        Args:
            Twc (numpy.ndarray): A 4x4 transformation matrix representing the pose of the camera in the world frame.
            pw (numpy.ndarray): A 3D point in the world coordinates.

        Returns:
            numpy.ndarray: The projected 3D point in the camera frame.
        """
        T_c_w = np.linalg.inv(Twc)
        return self.project3d(transform(T_c_w, pw))

    def project3dBatch(self, pts, margin=0.01):
        """
        Projects a batch of 3D points to 2D using the `project3d` method.

        Args:
            pts (list): A list of 3D points to be projected.
            margin (float, optional): A margin value for the projection. Defaults to 0.01.

        Returns:
            list: A list of 2D points resulting from the projection of the input 3D points.
        """
        res = []
        for pt in pts:
            res.append(self.project3d(pt))
        return res

    def project3DWorldBatch(self, Twc, pws):
        """
        Projects a batch of 3D world points into the camera frame.

        Args:
            Twc (numpy.ndarray): A 4x4 transformation matrix representing the pose of the camera in the world frame.
            pws (numpy.ndarray): An array of 3D points in the world frame to be projected.

        Returns:
            numpy.ndarray: An array of 2D points in the image plane after projection.
        """
        T_c_w = np.linalg.inv(Twc)
        pcs = np.array([transform(T_c_w, v) for v in pws])
        return self.project3dBatch(pcs)

    
    @staticmethod
    def filterInvisible(us, pts):
        """
        Filters out invisible points from the given lists.

        This method takes two lists: `us` and `pts`. It filters out the elements
        in `us` that are `None` and returns the corresponding elements from `pts`.

        Args:
            us (list): A list of elements where some elements can be `None`.
            pts (list): A list of points corresponding to the elements in `us`.

        Returns:
            tuple: A tuple containing two numpy arrays:
                - The first array contains the non-None elements from `us`.
                - The second array contains the corresponding points from `pts`.
        """
        viz_us = np.array([v for v in us if v is not None])
        viz_pts = np.array([pts[i] for i, v in enumerate(us) if v is not None])
        return viz_us, viz_pts
    
    def initializeBinnedFeatures(self):
        """
        Initializes the binned features array.

        This method creates a numpy array of zeros with a size determined by 
        the square of the number of bins per dimension. The array is stored 
        in the `binned_features` attribute and is of type int32.
        """
        self.binned_features = np.zeros((self.num_bins_per_dim*self.num_bins_per_dim), dtype=np.int32)

    """
    Create a test camera instance with predefined parameters.

    Returns:
        Camera: A Camera object initialized with test parameters.
    """
    @staticmethod
    def createTestCam():
        cam = Camera(407.424437, 407.504883, 340.911041, 240.253188, 752, 480)
        return cam


# if __name__ == "__main__":
    # from scipy.spatial.transform import Rotation

    # cam = Camera.createTestCam()

    # print("The horizontal and vertical FoVs are {0} and {1}".format(cam.half_hfov, cam.half_vfov))

    # N = 500
    # pts = np.zeros((N, 3))
    # pts[:, 0] = np.random.uniform(-5, 5, N)
    # pts[:, 1] = np.random.uniform(-5, 5, N)
    # pts[:, 2] = np.random.uniform(1.0, 2.0, N)

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(121, projection='3d')
    # ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], alpha=0.1)
    
    # ax.auto_scale_xyz([-5, 5], [-5, 5], [-5, 5])

    # T = randomTranformation()

    # pplotCoordinateFrame3d(T, ax)
    # ax.scatter(T[0, 3], T[1, 3], T[2, 3], 'x')

    # pxs = cam.project3dBatch(pts)
    # pxs = cam.project3DWorldBatch(T, pts)
    
    # viz_pxs, viz_pts = cam.filterInvisible(pxs, pts)
    # ax.scatter(viz_pts[:, 0], viz_pts[:, 1], viz_pts[:, 2], c='r')
    # ax = fig.add_subplot(122)
    # ax.scatter(viz_pxs[:, 0], viz_pxs[:, 1])
    # plt.show()