'''
The script is to evaluate sscore of all trainset input data

Author: Yefan Zhou
'''
import numpy as np
from compute_prediction import silhouette_score



def main():
    ptcloud_dismatrix = np.load("/home/../../public/zyf/ECCV2020/distance_matrix/GT_Trainset_dis_matrix.npy")
    print(ptcloud_dismatrix.shape)
    ss = silhouette_score(ptcloud_dismatrix)
    print("SScore:", ss)

if __name__ == "__main__":
    main()