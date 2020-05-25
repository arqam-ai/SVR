# This is a code repo for ECCV rebuttal  

## Clustering Method to infer 3D shape (adapted from What3D)  

  What3D paper: https://arxiv.org/abs/1905.03678   
  What3D dataset: https://drive.google.com/drive/folders/1OmL-EFCB-I_YhvvZ_DHRpLbFYg0vYf8G  

------

  **May 24th**:   
  Implemented the Clustering method based on the cluster assignment given by What3D author  
  To run the script, place the *What3D* folder What3D outside of *ECCV2020_rebuttal*  

  ```bash
  cd model
  python clustering.py
  ```
  In order to quickly test if the code could run normally, you could modify *clustering.py* line 233 as:
  ```python
  trainset_idx = cluster_dic[cluster_idx][:3]
  ```

------

**May 25th:**

Implemented the calculation of silhouette score on What3D train set 

To run the script, assume you have placed the dataset at **May 24th**, 

```bash
cd eval
python eval_ss.py
```

 In order to quickly test if the code could run normally, you could modify *eval_ss.py* [line91](https://github.com/YefanZhou/ECCV2020_rebuttal/blob/b3d4c368d168c282ca1a11654ec9333edcb83299/eval/eval_ss.py#L91) as:

```python
num_of_sample = int(0.001 *  num_of_train_instance)
```

##### faster silhouette score dataset:

downsampled pointcloud (1024 points to 256 points) locate at:   ***eval/faster_ss_data/testset_256points.npy***  shape = (10432, 256, 3)

corresponding sampling index locates at  ***eval/faster_ss_data/testset_256points_index.npy***

random sample the faster dataset to 1000 instances:  ***eval/faster_ss_data/testset_ptcloud_random1000.npy***  shape = (1000, 256, 3)

corresponding sampling index locates at ***eval/faster_ss_data/testset_ptcloud_random1000_index.npy***    