import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_active_direction(fig, start, dir, color='rgb(0, 0, 255)', opacity=1, length=1.5, row=1, col=1):
    end = start + length*dir
    # plot cam directions
    fig.add_trace(go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode='lines', line=dict(color=color, width=5), opacity=opacity), row=row, col=col)
    fig.add_trace(go.Cone(x=[end[0]], y=[end[1]], z=[end[2]], u=[dir[0]], 
                          v=[dir[1]], w=[dir[2]], opacity=opacity, colorscale=[(0, color), (1, color)], 
                          showscale=False, sizemode='scaled', sizeref=0.2, cmin=0, cmax=1), row=row, col=col)
    return fig

def plot_landmarks(fig, landmarks, row=1, col=1, color='green', opacity=0.3):
    fig.add_trace(go.Scatter3d(x=landmarks[:, 0], y=landmarks[:, 1], z=landmarks[:, 2], mode='markers', marker=dict(color=color, size=2, opacity=opacity)), row=row, col=col)
    return fig

def plot_path(fig, path, row=1, col=1, color='red', opacity=0.3):
    fig.add_trace(go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], mode='lines', line=dict(color=color, width=5), opacity=opacity), row=row, col=col)
    return fig


def plot_cilyndrical_obstacles(fig, obstacles, max_z=8, row=1, col=1, color='yellow', opacity=0.3):

    def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
        z = np.linspace(0, height_z, 50)
        theta = np.linspace(0, 2*np.pi, 50)
        theta_grid, z_grid=np.meshgrid(theta, z)
        x_grid = radius*np.cos(theta_grid) + center_x
        y_grid = radius*np.sin(theta_grid) + center_y
        return x_grid, y_grid, z_grid
    
    for idx in range(len(obstacles)):
        Xc, Yc, Zc = data_for_cylinder_along_z(obstacles[idx, 0], obstacles[idx, 1], obstacles[idx, 2], max_z)
        fig.add_trace(go.Surface(x=Xc, y=Yc, z=Zc, colorscale=[(0, 'yellow'), (1, 'orange')], opacity=opacity), row=row, col=col)
    
    return fig

def plot_camera_frustum(fig, fov, aspect_ratio, near, far, camera_position, camera_orientation, row=1, col=1, color='blue'):
    # calculate frustum geometry
    tan_half_fov = np.tan(np.radians(fov / 2))
    near_height = 2 * tan_half_fov * near
    near_width = near_height * aspect_ratio
    far_height = 2 * tan_half_fov * far
    far_width = far_height * aspect_ratio

    # define frustum corners in camera coordinates
    near_corners_camera = np.array([
        [-near_width / 2, -near_height / 2, near],
        [near_width / 2, -near_height / 2, near],
        [near_width / 2, near_height / 2, near],
        [-near_width / 2, near_height / 2, near],
    ])

    far_corners_camera = np.array([
        [-far_width / 2, -far_height / 2, far],
        [far_width / 2, -far_height / 2, far],
        [far_width / 2, far_height / 2, far],
        [-far_width / 2, far_height / 2, far],
    ])

    # transform frustum corners to world coordinates using the rotation matrix
    near_corners_world = np.dot(near_corners_camera, camera_orientation.T) + camera_position
    far_corners_world = np.dot(far_corners_camera, camera_orientation.T) + camera_position

    # plot frustum lines
    for i in range(4):
        fig.add_trace(go.Scatter3d(x=[near_corners_world[i, 0], far_corners_world[i, 0]],
                                   y=[near_corners_world[i, 1], far_corners_world[i, 1]],
                                   z=[near_corners_world[i, 2], far_corners_world[i, 2]],
                                   mode='lines',
                                   line=dict(color=color)), row=row, col=col)

    # plot lines connecting near and far corners
    for i in range(4):
        fig.add_trace(go.Scatter3d(x=[near_corners_world[i, 0], near_corners_world[(i + 1) % 4, 0]],
                                   y=[near_corners_world[i, 1], near_corners_world[(i + 1) % 4, 1]],
                                   z=[near_corners_world[i, 2], near_corners_world[(i + 1) % 4, 2]],
                                   mode='lines',
                                   line=dict(color=color)), row=row, col=col)
        fig.add_trace(go.Scatter3d(x=[far_corners_world[i, 0], far_corners_world[(i + 1) % 4, 0]],
                                   y=[far_corners_world[i, 1], far_corners_world[(i + 1) % 4, 1]],
                                   z=[far_corners_world[i, 2], far_corners_world[(i + 1) % 4, 2]],
                                   mode='lines',
                                   line=dict(color=color)), row=row, col=col)

    return fig




