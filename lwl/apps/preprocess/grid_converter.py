import numpy as np
import argparse
import pickle
import os

def quat_is_good(qx, qy, qz, qw):
    """
    Checks if the given quaternion components form a unit quaternion.

    A unit quaternion has a norm (magnitude) of 1. This function calculates
    the norm of the quaternion and checks if it is approximately equal to 1
    within a tolerance.

    Parameters:
    qx (float): The x component of the quaternion.
    qy (float): The y component of the quaternion.
    qz (float): The z component of the quaternion.
    qw (float): The w component of the quaternion.

    Returns:
    bool: True if the quaternion is a unit quaternion, False otherwise.
    """
    unit_quat_norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    if(not np.testing.assert_approx_equal(unit_quat_norm, 1, 7)):
        return True
    return False

def parse_active_grid(filename):
    """
    Parses an active grid file and returns a dictionary representation of the grid.

    Args:
        filename (str): The path to the file containing the grid data.

    Returns:
        dict: A dictionary where each key is an index (str) and each value is a dictionary with the following keys:
            - "num" (int): The number of directions.
            - "pose" (np.ndarray): A numpy array representing the camera pose (x, y, z).
            - "directions" (list): A list of numpy arrays representing the quaternion directions.
            - "hits" (list): A list of integers representing the number of hits.
    """
    grid = dict()
    with open(filename, 'r') as infile:
        bucket, directions, hits = None, None, None
        line_num = 0
        for line in infile:
            if(line_num == 0):
                # append array idx, num directions and camera pose
                idx, num, cam_pos_x, cam_pos_y, cam_pos_z = line.split()
                bucket = dict()
                bucket["num"] = int(num)
                bucket["pose"] = np.array([float(cam_pos_x), float(cam_pos_y), float(cam_pos_z)])
                directions, hits = list(), list()
                line_num = 1
                continue
            try: # we need to make sure line contains a quaternion
                # if(bucket["num"] > 0):
                num_hits, qx, qy, qz, qw = line.split()
                is_good = quat_is_good(float(qx), float(qy), float(qz), float(qw))
                if(is_good):
                    hits.append(int(num_hits))
                    directions.append(np.array([float(qx), float(qy), float(qz), float(qw)]))
                    line_num += 1
            except: # add pose, if not directions, there might not be directions 
                idx, num, cam_pos_x, cam_pos_y, cam_pos_z = line.split()
                bucket = dict()
                bucket["num"] = int(num)
                bucket["pose"] = np.array([float(cam_pos_x), float(cam_pos_y), float(cam_pos_z)])
                directions, hits = list(), list()
                line_num = 1
                continue   
            # append to dict each element
            if(line_num > int(num)):
                bucket["directions"] = directions
                bucket["hits"] = hits
                grid[idx] = bucket
                line_num = 0
    return grid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_file', type=str, help='Convert a file containing the voxel grid from txt to pickle or vice versa', required=True)
    parser.add_argument('--ext', type=str, help='output extension must be .txt or .pickle', default='.pickle')
    args = parser.parse_args()
    

    if(args.ext == 'pickle'):
        grid = parse_active_grid(args.grid_file)
        for k, v in grid.items():
            for d in v["directions"]:
                is_good = quat_is_good(float(d[0]), float(d[1]), float(d[2]), float(d[3]))
                if(not is_good):
                    print("quaternion direction not good! ", d)
                    exit(0)

        print("processed buckets num {}".format(k))
        name, ext = os.path.splitext(args.grid_file)
        pickle_filename = name + '.pickle'
        with open(pickle_filename, 'wb') as file:
            pickle.dump(grid, file)
    elif(args.ext == '.txt'):
        file = open(args.grid_file, 'rb')
        grid = pickle.load(file)
        name, ext = os.path.splitext(args.grid_file)
        txt_filename = name + '.txt'
        print(txt_filename)
        with open(txt_filename, 'w') as file:
            for k, v in grid.items():
                # 0 {'num': 1, 'pose': array([ 0.        , -0.50647145, -0.29473962]), 'directions': [array([ 0.36360095, -0.49043973,  0.70360327, -0.36360095])], 'hits': [247.596865724696]}
                print(k, v)
                
                line = str(k) + ' ' + str(v['num']) + ' ' + str(v['pose'][0]) + ' ' + str(v['pose'][1]) + ' ' + str(v['pose'][2]) + '\n'
                # print(v['directions'])
                line += str(int(v['hits'][0])) + ' ' + str(v['directions'][0][0]) + ' ' + str(v['directions'][0][1]) + ' ' + str(v['directions'][0][2]) + ' ' + str(v['directions'][0][3]) + '\n'
                file.write(line)
    




   