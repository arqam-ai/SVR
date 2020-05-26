"""
Author: Yiru Shen
Purpose: read block distance matrix and convert to complete matrix
"""
import numpy as np
import glob

def main():
    files = glob.glob("Pred_DM_partial/*.npy")
    files.sort()
    print(len(files))

    N = 10432
    distance_matrix = np.empty((N, N), dtype=np.float)
    for idx, f in enumerate(files):
        start_index, end_index = int(f.split("_")[5]), int(f.split("_")[6])
        curr_dm = np.load(f)
        distance_matrix[start_index:end_index,:] = curr_dm[start_index:end_index,:]
        # for i in range(start_index, end_index):
        #     for j in range(i + 1, N):
        #         distance_matrix[i, j] = curr_dm[i, j]
    print("Is upper diag: {}".format(np.allclose(distance_matrix, np.triu(distance_matrix))))
    distance_matrix = distance_matrix + distance_matrix.T - np.diag(np.diag(distance_matrix))
    print("Is symm: {}".format(np.allclose(distance_matrix, distance_matrix.T)))
    print("Saving distance matrix ..............................................")
    np.save("distance_matrix/Pred_all_distance_matrix.npy", distance_matrix)

if __name__ == "__main__":
    main()