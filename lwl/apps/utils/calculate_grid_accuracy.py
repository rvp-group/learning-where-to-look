import argparse
import numpy as np
from tqdm import tqdm

from lwl.apps.utils.general_utils import *

def calculate_accuracy(errors_t, errors_r, max_err_t=10, max_err_r=20):
    total_num = len(errors_t)
    errors_t = np.asarray(errors_t)
    errors_r = np.asarray(errors_r)

    assert total_num == len(errors_r)

    errors_t = np.nan_to_num(errors_t, copy=True, nan=max_err_t, posinf=max_err_t, neginf=max_err_t) # 10 meters above the threshold error
    errors_r = np.nan_to_num(errors_r, copy=True, nan=max_err_r, posinf=max_err_r, neginf=max_err_r) # 20 deg above the threshold error

    # All conditions: (0.25m, 2°) / (0.5m, 5°) / (5m, 10°)
    # https://openaccess.thecvf.com/content_cvpr_2018/papers/Sattler_Benchmarking_6DOF_Outdoor_CVPR_2018_paper.pdf
    
    err_t_thresholds = {"very_high_precision" : 0.05, "high_precision" : 0.25, "medium_precision" : 0.5, "coarse_precision" : 5}
    err_r_thresholds = {"very_high_precision" : 0.4, "high_precision" : 2, "medium_precision" : 5, "coarse_precision" : 10}
    accuracies = {"very_high_precision" : 0, "high_precision" : 0, "medium_precision" : 0, "coarse_precision" : 0}
    
    for errt, errr, in zip(errors_t, errors_r):
        for (k, tt), tr in zip(err_t_thresholds.items(), err_r_thresholds.values()):
            if(errt < tt and errr < tr):
                accuracies[k] += 1
    
    # assert fail_r == fail_t
    # total_without_nans = total_num - fail_t

    results = dict()
    # failure_rate, success_rate = fail_t/total_num, total_without_nans/total_num
    results["total_num"] = total_num
    # results["succes_rate"] = success_rate
    for i, (k, v) in enumerate(accuracies.items()):
        results[i] = (k, err_t_thresholds[k], err_r_thresholds[k], v/total_num)
    results["error_median"] = (np.nanmedian(errors_t), np.nanmedian(errors_r))
    return results

def calculate_accuracy_with_nan(errors_t, errors_r):
    total_num = len(errors_t)
    errors_t = np.asarray(errors_t)
    errors_r = np.asarray(errors_r)

    fail_t = np.isnan(errors_t).sum()
    fail_r = np.isnan(errors_r).sum()

    # All conditions: (0.25m, 2°) / (0.5m, 5°) / (5m, 10°)
    # https://openaccess.thecvf.com/content_cvpr_2018/papers/Sattler_Benchmarking_6DOF_Outdoor_CVPR_2018_paper.pdf
    
    err_t_thresholds = {"very_high_precision" : 0.05, "high_precision" : 0.25, "medium_precision" : 0.5, "coarse_precision" : 5}
    err_r_thresholds = {"very_high_precision" : 0.4, "high_precision" : 2, "medium_precision" : 5, "coarse_precision" : 10}
    accuracies = {"very_high_precision" : 0, "high_precision" : 0, "medium_precision" : 0, "coarse_precision" : 0}
    
    for errt, errr, in zip(errors_t, errors_r):
        for (k, tt), tr in zip(err_t_thresholds.items(), err_r_thresholds.values()):
            if(errt < tt and errr < tr):
                accuracies[k] += 1
    
    assert fail_r == fail_t
    total_without_nans = total_num - fail_t

    results = dict()
    failure_rate, success_rate = fail_t/total_num, total_without_nans/total_num
    results["total_num"] = total_num
    results["succes_rate"] = success_rate
    for i, (k, v) in enumerate(accuracies.items()):
        results[i] = (k, err_t_thresholds[k], err_r_thresholds[k], v/total_without_nans)
    results["error_median"] = (np.nanmedian(errors_t), np.nanmedian(errors_r))
    return results

def get_accuracies_of_n_elements(data, original_dict, n, modality):
    errors_t, errors_r = list(), list()
    for key, value_list in original_dict.items():
        if(n > len(value_list)):
            print("cannot evaluate multiple directions for bucket best num {} is less than required {}".format(len(value_list), n))
            return errors_t, errors_r
        values = value_list[:n]
        
        if(modality=='random'):
            # pick randomly
            start_index = np.random.randint(0, len(value_list) - n + 1)
            values = value_list[start_index:start_index + n]

        for idx, pred in values:
            errors_t.append(data[idx]['err_t'].item())
            errors_r.append(data[idx]['err_r'].item())
    assert len(errors_t) == len(original_dict.keys())*n

    return errors_t, errors_r

def evaluate_data(dirs):
    errors_t, errors_r = list(), list() 
    skipped = 0
    for d in tqdm(dirs, desc="pre-processing buckets"):    
        file_path = get_single_filename(d, 'pickle')
        bucket = load_pickle(file_path)
        # pose = bucket['pose']
        viewing_directions_indices = [key for key in bucket.keys() if str(key).isdigit()]
        # if(viewing_directions_indices
        # for idx_viewpoint in viewing_directions_indices:
        #     if(len(bucket[idx_viewpoint]["view_data"]) > 0):
        #         errors_t.append(bucket[idx_viewpoint]["err_t"])
        #         errors_r.append(bucket[idx_viewpoint]["err_r"])
        errors_t.append(bucket[viewing_directions_indices[0]]["err_t"])
        errors_r.append(bucket[viewing_directions_indices[0]]["err_r"])
    
    errors = calculate_accuracy(errors_t, errors_r)    
    
    print("total num of samples {}".format(errors["total_num"]))
    print("percentage cases of success {:.3f}".format(errors["succes_rate"]))
    print("localization benchmark results: ")
    integer_keys = [key for key in errors.keys() if isinstance(key, int)]
    for k in integer_keys:
        (k, err_t_thresholds, err_r_thresholds, valid_ratio) = errors[k]
        print("\t{} ({} m, {} deg) : {:.3f}".format(k, err_t_thresholds, err_r_thresholds, valid_ratio)) 
    err_t_median, err_r_median = errors["error_median"]   
    print("median t error {:.3f} m and r error {:.3f} deg".format(err_t_median, err_r_median))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='viewpoint evaluator')
    parser.add_argument('--data_path', type=str, required=True, help='path to your data file')
    args = parser.parse_args()

    ############### start processing folders ###############
    # get root path and path to each folder representing a camera bucket in the grid world
    ordered_dirs = list_and_order_directories(args.data_path)

    # get num of samples for dir, assuming they are the same
    file_path_tmp = get_single_filename(ordered_dirs[0], 'pickle')
    bucket_tmp = load_pickle(file_path_tmp)
    viewing_directions_indices_tmp = [key for key in bucket_tmp.keys() if str(key).isdigit()]
    num_samples_per_bucket = len(viewing_directions_indices_tmp)

    evaluate_data(ordered_dirs)