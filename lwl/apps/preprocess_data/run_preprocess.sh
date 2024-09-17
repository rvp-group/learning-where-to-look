# train
# python3 preprocess.py --data_path /media/ldg/T71/MH3D_10_scene/ --positive_percentage 0.5 --rows 480 --cols 640 --num_bin_per_dim 30 --output_data /media/ldg/T71/matterport_data_50 >& /media/ldg/T71/matterport_data_50_out.txt;
# python3 preprocess.py --data_path /media/ldg/T71/MH3D_10_scene/ --positive_percentage 0.4 --rows 480 --cols 640 --num_bin_per_dim 30 --output_data /media/ldg/T71/matterport_data_40 >& /media/ldg/T71/matterport_data_40_out.txt;
# python3 preprocess.py --data_path /media/ldg/T71/MH3D_10_scene/ --positive_percentage 0.3 --rows 480 --cols 640 --num_bin_per_dim 30 --output_data /media/ldg/T71/matterport_data_30 >& /media/ldg/T71/matterport_data_30_out.txt;
# python3 preprocess.py --data_path /media/ldg/T71/MH3D_10_scene/ --positive_percentage 0.6 --rows 480 --cols 640 --num_bin_per_dim 30 --output_data /media/ldg/T71/matterport_data_60 >& /media/ldg/T71/matterport_data_60_out.txt;

# test
python3 preprocess.py --data_path /media/ldg/T71/MH3D_2_test/ --positive_percentage 0 --rows 480 --cols 640 --num_bin_per_dim 30 --output_data /media/ldg/T71/matterport_data_test
