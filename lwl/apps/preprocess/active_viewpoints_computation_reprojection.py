import argparse
import yaml
import sys
import numpy as np

from lwl.pyactivegrid import ActiveGrid
from lwl.apps.utils.colmap.colmap_read_write_model import parse_landmarks

parser = argparse.ArgumentParser(description='active camera grid calculation based on visibility')
parser.add_argument('--landmarks', type=str, help='supports colmap type, or simply vector of 3d points')
parser.add_argument('--config_path', type=str, help='path to yaml configuration file', required=True)
parser.add_argument('--output_file', type=str, help='path to output grid', required=True)
parser.add_argument('--mesh_path', type=str, help='use mesh to calculate minimum and maximum bounds of 3d map along each axis, if not uses the sparse model which might contain errors in landmarks position')
parser.add_argument('--enable_viz', action='store_true', help='[DEBUG] enable voxel/locations visualization')
args = parser.parse_args()

infile = open(args.config_path, 'r')
configuration_file = yaml.safe_load(infile)

# getting some parameters
f_length           = configuration_file["focal-length"]
rows               = configuration_file["rows"]
cols               = configuration_file["cols"]
num_samples        = configuration_file["num-samples"]
max_mapping_error  = configuration_file["colmap-error"]
max_directions_to_keep  = configuration_file["num-max-directions"]
bucket_ratio  = configuration_file["bucket-ratio"]

# getting principal points for virtual world
cx = cols/2-0.5
cy = rows/2-0.5

# adding some depth values to filer reprojections
min_depth = 0.001
max_depth = sys.float_info.max

landmarks, errors, indices = parse_landmarks(args.landmarks, max_mapping_error)
errors_reshaped = np.expand_dims(errors, axis=1)
indices_reshaped = np.expand_dims(indices, axis=1)

x_max = np.max(landmarks[:, 0])
y_max = np.max(landmarks[:, 1])
z_max = np.max(landmarks[:, 2])

x_min = np.min(landmarks[:, 0])
y_min = np.min(landmarks[:, 1])
z_min = np.min(landmarks[:, 2])

if args.mesh_path is not None:
    try:
        import open3d as o3d
        print("Open3D version: ", o3d.__version__)
        mesh = o3d.io.read_triangle_mesh(args.mesh_path)
        vertices = np.asarray(mesh.vertices)
        # calculate the maximum and minimum values along each axis
        x_max, y_max, z_max = np.max(vertices, axis=0)
        x_min, y_min, z_min = np.min(vertices, axis=0)
        print("using mesh to calculate precisely bounds of map")
    except ImportError:
        print("Open3D is not installed, install it using: pip install open3d or don't use the mesh_path argument")
        exit(-1)
else:
    print("WARNING: voxelization might not be correct given outliers in SfM model, suggested to input the mesh file")
    
min_grid_pos = np.array([x_min, y_min, z_min])
max_grid_pos = np.array([x_max, y_max, z_max]) 

# precalculating grid dim and bucket extents
grid_dimensions = np.abs(max_grid_pos) + np.abs(min_grid_pos)
bucket_extents = np.array(grid_dimensions / bucket_ratio)
max_grid_pos += bucket_extents

# calculating again using updated max grid value, otherwise we always miss some part of the map
grid_dimensions = np.abs(max_grid_pos) + np.abs(min_grid_pos)
bucket_extents = np.array(grid_dimensions / bucket_ratio)
print("overriding min and max grid positions, inference from data: {} {}".format(min_grid_pos, max_grid_pos))
print("bucket extents: {}".format(bucket_extents))

sparse = np.concatenate((landmarks, errors_reshaped, indices_reshaped), axis=1)

active_grid = ActiveGrid()
active_grid.setCamera(rows, cols, min_depth, max_depth, f_length, f_length, cx, cy)
active_grid.setNumSamples(num_samples)
active_grid.setSparseLandmarks(sparse)
active_grid.setBucketExtents(bucket_extents)
active_grid.setGridMinAndMax(min_grid_pos, max_grid_pos)
active_grid.setGridDimension(grid_dimensions)
active_grid.compute()

grid_file = open(args.output_file, "w")
for i in range(active_grid.size()):
    pose = active_grid.getPose(i)
    dirs = active_grid.getBestViewingDirections(i, max_directions_to_keep)
    all_hits = active_grid.getBestViewingHits(i, max_directions_to_keep)
    line = str(i) + " " + str(len(all_hits)) + " " + str(pose[0]) + " " + str(pose[1]) + " " + str(pose[2]) + "\n"
    for (dir, hits) in zip(dirs, all_hits):
        line += str(hits) + " " + str(dir[0]) + " " + str(dir[1]) + " " + str(dir[2]) + " " + str(dir[3]) + "\n"
    grid_file.write(line)
grid_file.close()

if(args.enable_viz):
    locations = list()
    for i in range(active_grid.size()):
        locations.append(active_grid.getPose(i))

    locations = np.asarray(locations)
    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(locations)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh, points], window_name="camera locations")





