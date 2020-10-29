import sys
import tqdm
import open3d as o3d
import numpy as np
sys.path.append("../")
from utils.utils import normalize_bbox

CUBE_SIDE_LEN = 1.0
threshold_list = [CUBE_SIDE_LEN/200, CUBE_SIDE_LEN/100,
                    CUBE_SIDE_LEN/50, CUBE_SIDE_LEN/20,
                    CUBE_SIDE_LEN/10, CUBE_SIDE_LEN/5]
BBOX = np.array([[[0, 0, 0], [1.0, 1.0, 1.0]]])

def calculate_fscore(gt, pr, th =0.01):
    '''
    Calculates the F-score between two point clouds with the corresponding threshold value.
    gt : open3d.geometry.PointCloud
    pr : open3d.geometry.PointCloud
    typing.Tuple[float, float, float]
    '''
    d1 = gt.compute_point_cloud_distance(pr)
    d2 = pr.compute_point_cloud_distance(gt)

    #d1 = open3d.compute_point_cloud_to_point_cloud_distance(gt, pr)
    #d2 = open3d.compute_point_cloud_to_point_cloud_distance(pr, gt)
    
    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(len(d2))
        precision = float(sum(d < th for d in d1)) / float(len(d1))

        if recall+precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0

    return fscore, precision, recall

def f_score_list(th, pred_set, gt_set, normalize):
    """
    Params:
    ----------
    th       : float
    pred_set : numpy.array (sample_num, pt_num, 3)
    gt_set   : numpy.array (sample_num, pt_num, 3)
    normalize: str 
        option: ["unitL2ball", "bbox"]

    Returns:
    ----------
    f_list  : python.list
    p_list  : python.list
    r_list  : python.list
    """
    if normalize == "unitL2ball":
        pass
    elif normalize == "bbox":
        pred_set = normalize_bbox(pred_set, BBOX, isotropic=True)
        gt_set = normalize_bbox(gt_set, BBOX, isotropic=True)
    f_list = []
    p_list = []
    r_list = []
    sample_num = pred_set.shape[0]
    
    for idx in range(sample_num):
        gt = o3d.geometry.PointCloud()
        gt.points = o3d.utility.Vector3dVector(gt_set[idx])
        pr = o3d.geometry.PointCloud()
        pr.points = o3d.utility.Vector3dVector(pred_set[idx])
        f, p, r = calculate_fscore(gt, pr, th)
        f_list.append(f)
        p_list.append(p)
        r_list.append(r)
    
    return (f_list, p_list, r_list)


def f_score(th, pred_set, gt_set, out_path):
    """
    Params:
    ----------
    th:       python.list
    pred_set: numpy.array
    gt_set:   numpy.array
    out_path: str

    Returns:
    ----------
    thre_mean: python.dic
    """
    CUBE_SIDE_LEN = 1.0
    shapenum = pred_set.shape[0]
    if th is None:
        threshold_list = [CUBE_SIDE_LEN/200, CUBE_SIDE_LEN/100,
                    CUBE_SIDE_LEN/50, CUBE_SIDE_LEN/20,
                    CUBE_SIDE_LEN/10, CUBE_SIDE_LEN/5]
    else:
        threshold_list = [th]
    thre_mean = {'f':[], 'p':[], 'r':[]}
    for th in tqdm.tqdm(threshold_list, total = len(threshold_list), desc="Threshold"):
        score = {'f':[],'p':[],'r':[]}
        for idx in tqdm.tqdm(range(shapenum),total = len(range(shapenum)), desc="Instances"):
            gt = o3d.geometry.PointCloud()
            gt.points = o3d.utility.Vector3dVector(gt_set[idx])
            pr = o3d.geometry.PointCloud()
            pr.points = o3d.utility.Vector3dVector(pred_set[idx])
            f, p, r = calculate_fscore(gt, pr, th = th)
            score['f'].append(f)
            score['p'].append(p)
            score['r'].append(r)

        thre_mean['f'].append(sum(score['f'])/len(score['f']))
        thre_mean['p'].append(sum(score['p'])/len(score['p']))
        thre_mean['r'].append(sum(score['r'])/len(score['r']))

    return threshold_list, thre_mean
    