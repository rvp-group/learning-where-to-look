import torch
from torch.utils.data import DataLoader

import argparse, sys

from lwl.apps.training.data_loader import DataMLPTrain
from lwl.apps.training.trainer import MLPTrainer, MODEL_NAME
from lwl.apps.training.standardize_data_5d_features import make_set

from lwl.apps.utils.general_utils import *
from lwl.apps.utils.seed import *
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute active map')

    parser.add_argument('--train_data_path', type=str, required=True, help='Path to your train data file, important for standardizing the data (mean and std)')
    parser.add_argument('--evaluate_data_path', type=str, required=True, help='Path to your raw data you want to use to compute the active map')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to torch model, start from existing model if available')
    parser.add_argument('--max_features', type=int, help='Number of features to use, if num bin used is num_bins*num_bins', default=900)

    # enable visualization flag, set to True if passed
    parser.add_argument('--enable_viz', action='store_true', help='Enable visualization of active map')
    parser.add_argument('--animate', action='store_true', help='Enable animation of active map, from lower to higher probabilites predicted')
    parser.add_argument('--config_path', type=str, help='Path to YAML configuration file, if not provided, cameras will be plotted as directions, not frustum')
    parser.add_argument('--landmarks', type=str, required=False, help='Path to SFM model with 3D sparse landmarks')

    args = parser.parse_args()

    # Check if visualization is enabled and landmarks/config_path are required
    if args.enable_viz:
        if args.config_path is None:
            print("Error: --config_path is required when visualization is enabled.")
            sys.exit(1)
        if args.landmarks is None:
            print("Error: --landmarks is required when visualization is enabled.")
            sys.exit(1)


    # check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device {}".format(device))

    dataset = load_dataset(args.train_data_path)
    normalizer = (dataset["mean"], dataset["std"])
    print("loaded data from {} | using this just to standardize dataset to evaluate with | mean {} std {}".format(args.train_data_path, normalizer[0], normalizer[1]))
       
    raw_test_dataset = load_dataset(args.evaluate_data_path)
    print("loaded raw evaluation data from {}".format(args.evaluate_data_path))
    test_dataset = dict()      
    test_dataset = [make_set(sample) for sample in raw_test_dataset.values()]
    print("completed loaded and preprocessing")
    test_data = DataMLPTrain(test_dataset, device, max_features=args.max_features, normalizer=normalizer, is_training=False) 
    
    NUM_WORKERS = 1
    PIN_MEMORY = False
    BATCH_SIZE = 128 # only to speed up inference
    
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # initialize the MLP model
    print("loading model from {}".format(args.model_dir))

    _, model, _ = MLPTrainer.start_from_existing_model(args.model_dir, MODEL_NAME)
    print("model ", model)

    model.eval() 
    pred_test = list()               
    for test_batch in test_loader: 
        pred = model(test_batch['input'].to(device=device, non_blocking=PIN_MEMORY))
        pred_test.append(pred.squeeze())
        
    predictions_list_of_list = [t.tolist() for t in pred_test]
    predictions = [item for sublist in predictions_list_of_list for item in sublist]
    best_viewpoints_per_bucket = dict()
    # order all_preds per bucket
    for idx, pred in enumerate(predictions):
        pose_idx = raw_test_dataset[idx]['bucket_id']
        if (pose_idx not in best_viewpoints_per_bucket.keys()):
            best_viewpoints_per_bucket[pose_idx] = list()
        pose = raw_test_dataset[idx]['pose']
        quat = raw_test_dataset[idx]['quat']
        pts_3d = raw_test_dataset[idx]['pts_3d']
        best_viewpoints_per_bucket[pose_idx].append((idx, pred, pose, quat, pts_3d))

    best_direction_dict = {key: sorted(value, key=lambda x: x[1], reverse=True) for key, value in best_viewpoints_per_bucket.items()}
    # if not plotting dump this dictionary here
    if not args.enable_viz:
        import pickle
        with open('active_map.pkl', 'wb') as f:
            pickle.dump(best_direction_dict, f)
        sys.exit(0)

    # parse configuration
    best_direction_dict_ascent = {key: sorted(value, key=lambda x: x[1]) for key, value in best_viewpoints_per_bucket.items()}
    
    import yaml
    infile = open(args.config_path, 'r')
    configuration_file = yaml.safe_load(infile)
    
    # getting some parameters
    f_length           = configuration_file["focal-length"]
    rows               = configuration_file["rows"]
    cols               = configuration_file["cols"]
    max_mapping_error  = configuration_file["colmap-error"]

    import lwl.apps.utils.camera as c
    cam = c.Camera(fx=f_length, fy=f_length, cx=cols/2, cy=rows/2, rows=rows, cols=cols)
    hfov = cam.half_hfov * 2
    vfov = cam.half_vfov * 2

    from lwl.apps.utils.colmap.colmap_read_write_model import parse_landmarks
    landmarks, errors, indices = parse_landmarks(args.landmarks, max_mapping_error)

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    from lwl.apps.utils.plotting import plot_landmarks, plot_camera_frustum, plot_active_direction

    fig = make_subplots(
    rows=1, 
    cols=1,
    start_cell="top-left", 
    specs=[[{"type": "scatter3d"}]])


    # print(best_direction_dict)

    if(args.animate):
        import plotly.colors as colors
        def get_color_from_prob(prob, min_prob=0.0, max_prob=1.0, color_scale='YlGnBu'):
            norm_prob = (prob - min_prob) / (max_prob - min_prob)    
            color_scale_func = getattr(colors.sequential, color_scale)
            # print(color_scale_func)
            color_tuples = [tuple(map(int, color.replace('rgb(', '').replace(')', '').split(','))) for color in color_scale_func]
            color_float = colors.find_intermediate_color(color_tuples[0], color_tuples[-1], norm_prob)
            color = tuple(map(round, color_float))
            return 'rgb'+str(color)
            
        # animation plot
        from scipy.spatial.transform import Rotation
        animation_data = dict()
        LINE_LENGTH = 0.5
        for values in best_direction_dict_ascent.values():
            counter = 0
            for v in values:
                _, prob, pose, quat, pts_3d = v # this is best prediction
                if(counter not in animation_data.keys()):
                    animation_data[counter] = list()
                rot = Rotation.from_quat(quat)
                dir = rot.as_matrix()[0:3, 2]
                end = pose + LINE_LENGTH*dir
                color = get_color_from_prob(prob)
                color_pts = get_color_from_prob(prob, color_scale='Reds')
                animation_data[counter].append((prob, color, color_pts, pose, dir, end, pts_3d))
                counter += 1
                
        for prob, color, color_pts, pose, dir, end, pts_3d in animation_data[0]:
            fig.add_trace(go.Scatter3d(x=pts_3d[:, 0], y=pts_3d[:, 1], z=pts_3d[:, 2], mode='markers', marker=dict(color=color_pts, size=2, opacity=0.3)))
            fig.add_trace(go.Scatter3d(x=[pose[0], end[0]], y=[pose[1], end[1]], z=[pose[2], end[2]],
                                    mode='lines', line=dict(color=color, width=5), opacity=0.4))
            fig.add_trace(go.Cone(x=[end[0]], y=[end[1]], z=[end[2]], u=[dir[0]], 
                            v=[dir[1]], w=[dir[2]], opacity=0.4, colorscale=[(0, color), (1, color)], 
                            showscale=False, sizemode='scaled', sizeref=0.2, cmin=0, cmax=1))

        import tqdm
        # create frames for the animation
        frames = list()
        color_idx = 0
        for counter, values in tqdm.tqdm(animation_data.items(), desc='building animation'):
            frame_data = list()
            for prob, color, color_pts, pose, dir, end, pts_3d in values:
                frame_data.append(go.Scatter3d(x=pts_3d[:, 0], y=pts_3d[:, 1], z=pts_3d[:, 2], mode='markers', marker=dict(color=color_pts, size=2, opacity=0.3)))
                frame_data.append(go.Scatter3d(x=[pose[0], end[0]], y=[pose[1], end[1]], z=[pose[2], end[2]],
                                            mode='lines', line=dict(color=color, width=5), opacity=0.4))
                frame_data.append(go.Cone(x=[end[0]], y=[end[1]], z=[end[2]], u=[dir[0]], 
                                v=[dir[1]], w=[dir[2]], opacity=0.4, colorscale=[(0, color), (1, color)], 
                                showscale=False, sizemode='scaled', sizeref=0.2, cmin=0, cmax=1))
                color_idx += 1
            # frame_data.append(go.Scatter3d(x=landmarks[:, 0], y=landmarks[:, 1], z=landmarks[:, 2], mode='markers', marker=dict(color='green', size=2, opacity=0.3)))
            frames.append(go.Frame(data=frame_data, name=f'frame{counter}'))

        # add the frames to the figure
        fig.frames = frames
        
        
        # add play and pause buttons for the animation
        fig.update_layout(
            updatemenus=[
                {
                    'buttons': [
                        {
                            'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'mode': 'immediate'}],
                            'label': 'Play',
                            'method': 'animate'
                        },
                        {
                            'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                            'label': 'Pause',
                            'method': 'animate'
                        }
                    ],
                    'direction': 'down',
                    'pad': {'r': 10, 't': 10},
                    'showactive': False,
                    'type': 'dropdown',
                    'x': 0.1,
                    'xanchor': 'left',
                    'y': 1.1,
                    'yanchor': 'top'
                }
            ]
        )

        # 3D view    
        name = 'eye = (x:2, y:2, z:0.1)'
        camera = dict(
            eye=dict(x=2, y=2, z=0.1)
        )

        
        # X-Z plane 
        # name = 'eye = (x:0., y:2.5, z:0.)'
        # camera = dict(
        # eye=dict(x=0., y=2.5, z=0.)
        # )

        # X-Y plane 
        # name = 'eye = (x:0., y:0., z:2.5)'
        # camera = dict(
        #     eye=dict(x=0., y=0., z=2.5)
        # )

        fig.update_layout(scene_camera=camera, plot_bgcolor='white', showlegend=False, 
            scene = dict(xaxis = dict(nticks=4, range=[np.min(landmarks[:, 0]), np.max(landmarks[:, 0])],),
                        yaxis = dict(nticks=4, range=[np.min(landmarks[:, 1]), np.max(landmarks[:, 1])],),
                        zaxis = dict(nticks=4, range=[np.min(landmarks[:, 2]), np.max(landmarks[:, 2])],),))
        fig.update_scenes(aspectmode='data')
        fig.show()    
    else:
    # static plot
        print(best_direction_dict)
        probs, locations, quats = list(), list(), list()
        for values in best_direction_dict.values():
            _, prob, pos, quat = values[0] # this is best prediction
            probs.append(prob)
            locations.append(pos)
            quats.append(quat)

        from plotly.express.colors import sample_colorscale
        colors = sample_colorscale('YlGnBu', probs)
        print(colors)
        exit(0)
        
        from scipy.spatial.transform import Rotation
        fig = plot_landmarks(fig, np.asarray(locations), row=1, col=1, color='red', opacity=0.3)
        for (pose, quat, color) in zip(locations, quats, colors):
            rot = Rotation.from_quat(quat)
            # fig = plot_camera_frustum(fig, np.rad2deg(hfov), aspect_ratio=(cols / rows), near=0.05, far=0.3, camera_position=pose, camera_orientation=rot.as_matrix(), color=color)
            fig = plot_active_direction(fig, start=pose, dir=rot.as_matrix()[0:3, 2], length=0.5, row=1, col=1, opacity=0.4, color=color)

        
        fig.update_layout(showlegend=False)
        fig.update_scenes(aspectmode='data')
        fig.show()




