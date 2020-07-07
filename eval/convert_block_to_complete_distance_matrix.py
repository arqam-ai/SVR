"""
Author: Yiru Shen, Yefan Zhou
Purpose: read block distance matrix and convert to complete matrix
"""
import os
import sys
import numpy as np
import glob
import argparse

def main(args):
    # Get all the partial matrix
    files = glob.glob(os.path.join(args.partial_path, "*.npy"))
    files.sort()    
    start_index_list = []
    end_index_list = []

    # Initialize the matrix
    N = args.mat_size
    distance_matrix = np.empty((N, N), dtype=np.float)

    # Traverse through the files 
    for idx, f in enumerate(files):
        start_index, end_index = int(f.split("_")[5]), int(f.split("_")[6])
        start_index_list += [start_index]
        end_index_list += [end_index]
        # Value Transfer
        curr_dm = np.load(f)
        distance_matrix[start_index:end_index,:] = curr_dm[start_index:end_index,:]
    
    # Check if start/end index consistent
    start_index_list.sort()
    end_index_list.sort()
    if_consistent = True
    for i in range(1, len(start_index_list)):
        if_consistent = if_consistent and start_index_list[i] == end_index_list[i-1]
    if_consistent = if_consistent and end_index_list[-1] == N
    print("Is convert consistent: {}".format(if_consistent))

    # Check if distance matrix semi-identical
    print("Is upper diag: {}".format(np.allclose(distance_matrix, np.triu(distance_matrix))))
    distance_matrix = distance_matrix + distance_matrix.T - np.diag(np.diag(distance_matrix))
    print("Is symm: {}".format(np.allclose(distance_matrix, distance_matrix.T)))
    print("Saving distance matrix ..............................................")
    np.save(args.save_path, distance_matrix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('--partial_path', type=str, default='/home/../public/zyf/ECCV2020/distance_matrix/PC_CD_partial',
                        help='')
    parser.add_argument('--save_path', type=str, default='/home/../public/zyf/ECCV2020/distance_matrix/GT_Trainset_dis_matrix.npy',
                        help='')
    parser.add_argument('--mat_size', type=int, default=36757, 
                        help='')

    args = parser.parse_args(sys.argv[1:])
    args.script_folder = os.path.dirname(os.path.abspath(__file__))
    main(args)
