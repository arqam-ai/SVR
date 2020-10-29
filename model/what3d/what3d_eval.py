"""
    This script is from https://github.com/lmb-freiburg/what3d/blob/a182b688aee0f0c7bc8f16f1f278f2c22848ec85/util.py#L33
    Used for dense pointcloud(9000+) raw data from what3d paper in order to reproduce Oralce NN viewer center data mode f-score curve 
"""
import os
import sys
import numpy as np
sys.path.append("../../../")
import utils.what3d_util as util
from utils.f_score import calculate_fscore
from dataset.dataset import what3d_dataset_views
import tqdm
import open3d
import argparse
import random
import torch

CUBE_SIDE_LEN = 1.0

open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Error)

parser = argparse.ArgumentParser(description='F-score evaluation')
parser.add_argument('--pr_path', type=str, required=False, default="")
parser.add_argument('--out_path', type=str, default="")
parser.add_argument('--th', type=float, default=None)
args = parser.parse_args()


def f_score(args):
    if args.th is None:
        threshold_list = [CUBE_SIDE_LEN/200, CUBE_SIDE_LEN/100,
                        CUBE_SIDE_LEN/50, CUBE_SIDE_LEN/20,
                        CUBE_SIDE_LEN/10, CUBE_SIDE_LEN/5]
    else:
        threshold_list = [args.th]
    class_eval_stats = np.load(os.path.join("exp2", "class_eval_stats.npy"), allow_pickle=True).item()
    class_eval_index = class_eval_stats["index"]
    class_list = util.get_class_list()
    test_set_path = []
    train_set_path = []
    pred_set_path = []
    for cat in class_list:
        #os.mkdir(os.path.join(out_path, cat))
        test_model_list = util.get_class_models(cat, mode='test')
        train_model_list = util.get_class_models(cat, mode='train')
        
        for v in range(util.VIEW_COUNT):
            for i in range(len(test_model_list)):
                model = test_model_list[i]
                test_set_path.append(os.path.join(util.POINTS_PATH, cat, model, str(v) + ".ply"))
        class_train_path = []
        for v in range(util.VIEW_COUNT):
            for i in range(len(train_model_list)):
                model = train_model_list[i]
                class_train_path.append(os.path.join(util.POINTS_PATH, cat, model, str(v) + ".ply"))
        
        
        for pred_index in class_eval_index[cat]:
            if not isinstance(pred_index, int):
                pred_index = pred_index[0]
            pred_set_path.append(class_train_path[pred_index])

        train_set_path += class_train_path


    print(len(test_set_path))
    print(len(pred_set_path))
    num_instance = len(test_set_path)
    thre_mean = {'f':[], 'p':[], 'r':[]}
    for th in tqdm.tqdm(threshold_list, total = len(threshold_list), desc="Threshold"):
        score = {'f':[],'p':[],'r':[]}
        for idx in tqdm.tqdm(range(num_instance),total = num_instance, desc="Instances"):
            gt = open3d.io.read_point_cloud(test_set_path[idx])
            pr = open3d.io.read_point_cloud(pred_set_path[idx])
            f, p, r = calculate_fscore(gt, pr, th = th)
            score['f'].append(f)
            score['p'].append(p)
            score['r'].append(r)
        np.save(os.path.join("f_score", "score_{}.npy".format(th)), score)
        thre_mean['f'].append(sum(score['f'])/len(score['f']))
        thre_mean['p'].append(sum(score['p'])/len(score['p']))
        thre_mean['r'].append(sum(score['r'])/len(score['r']))

    print(threshold_list)
    print(thre_mean)

def f_score_perc(path, th):
    score = np.load(os.path.join(path, "f_score/score_{}.npy".format(th)),allow_pickle=True).item()
    f_score = score['f']
    print(sum(f_score)/len(f_score))
    perc = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    total_count = len(score['f'])
    score_count = [0, 0, 0, 0, 0, 0]
    for f_score in score['f']:
        for count_idx in range(len(score_count)):
            if f_score >= perc[count_idx]:
                score_count[count_idx] += 1

    plot_input = [counts/total_count for counts in score_count]
    print("f", plot_input)

    print(total_count)
    score_count = [0, 0, 0, 0, 0, 0]
    for p_score in score['p']:
        for count_idx in range(len(score_count)):
            if p_score >= perc[count_idx]:
                score_count[count_idx] += 1
    p_plot_input = [counts/total_count for counts in score_count]
    print("p", p_plot_input)

        
    score_count = [0, 0, 0, 0, 0, 0]
    for r_score in score['r']:
        for count_idx in range(len(score_count)):
            if r_score >= perc[count_idx]:
                score_count[count_idx] += 1
    r_plot_input = [counts/total_count for counts in score_count]
    print("r", r_plot_input)


#f_score_perc(args)








###Visualize the pointcloud to check consistent
# pred_set = np.load("pred_set.npy")
# gt_set = np.load("gt_set.npy")
# sample_index_list = random.sample(range(len(test_set_path)), 20)
# for order, sample_index in enumerate(sample_index_list):
#     gt_ptcloud1 = gt_set[sample_index]
#     gt_ptcloud2 = np.asarray(open3d.io.read_point_cloud(test_set_path[sample_index]).points)
#     pred_ptcloud = np.asarray(open3d.io.read_point_cloud(pred_set_path[sample_index]).points)
#     what3d_dataset_views.data_visualizer(torch.from_numpy(gt_ptcloud1), torch.from_numpy(gt_ptcloud2), torch.zeros(1, 3, 224, 224), "gt", "test_what3d_raw", order)
#     what3d_dataset_views.data_visualizer(torch.from_numpy(gt_ptcloud1), torch.from_numpy(pred_ptcloud), torch.zeros(1, 3, 224, 224), "pred", "test_what3d_raw", order)
            
