# This is a code repo for ECCV rebuttal  

## Clustering Method to infer 3D shape (adapted from What3D)  

  What3D paper: https://arxiv.org/abs/1905.03678   
  What3D dataset: https://drive.google.com/drive/folders/1OmL-EFCB-I_YhvvZ_DHRpLbFYg0vYf8G  

------

#####   **May 24th**:   

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

##### **May 25th:**

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

------

##### **May 26th**

1.  [link ](https://github.com/YefanZhou/ECCV2020_rebuttal/blob/master/eval/faster_eval_ss.py)  calculate silhouette score of randomly sampled test set ground truth point cloud (Num: 2000)  repeat the experiment for ten times,  code

```bash
cd eval
python faster_eval_ss.py
Output:
  Experiment No.1 point cloud silhouette: 0.06785896420478821
  Time:506.22960805892944 seconds
  Experiment No.2 point cloud silhouette: 0.07321267575025558
  Time:1009.3026542663574 seconds
  Experiment No.3 point cloud silhouette: 0.07592274248600006
  Time:1512.7354078292847 seconds
  Experiment No.4 point cloud silhouette: 0.07296927273273468
  Time:2020.5051062107086 seconds
  Experiment No.5 point cloud silhouette: 0.07694291323423386
  Time:2528.785288333893 seconds
  Experiment No.6 point cloud silhouette: 0.06453316658735275
  Time:3035.0339250564575 seconds
  Experiment No.7 point cloud silhouette: 0.07341325283050537
  Time:3538.3120381832123 seconds
  Experiment No.8 point cloud silhouette: 0.0679931789636612
  Time:4040.571636199951 seconds
  Experiment No.9 point cloud silhouette: 0.0679311752319336
  Time:4544.851239442825 seconds
  Experiment No.10 point cloud silhouette: 0.07102470844984055
  Time:5050.380216360092 seconds
```



1. [link](https://github.com/YefanZhou/ECCV2020_rebuttal/blob/master/eval/compute_prediction.py)  generate **Clustering Method** prediction distance matrx (10432 x 10432) based on cluster distance matrix.

   and calculate silhouette score of  **Clustering Method** prediction distance matrx (10432 x 10432) 

```bash
cd eval
python compute_prediction.py 
Output:
  normalize done Time : 0.995206 seconds
  paritition done Time : 245.385602 seconds
  silhouette done Time : 246.303075 seconds
  silhouette_score is 0.651192
```

------

##### **May 27th**

1. Farthest Point Sampling sample 2000 instances and then calculate silhouette score

   [FPS implementation](https://github.com/YefanZhou/ECCV2020_rebuttal/blob/759377ef500955409bf8237d4782df0ef700fa90/eval/subsample_selection_yefan.py#L170) [reference](https://flothesof.github.io/farthest-neighbors.html) 

   **To run the script:** 

```bash
cd eval
```

modify ***subsample_selection_yefan.py***  line 298 as 

```python
subsample_type = "FPS"
```

```bash
python subsample_selection_yefan.py
Output:
	experiment 0: silhouette score is 0.026271342397641534
	experiment 1: silhouette score is 0.02909122411897761
	experiment 2: silhouette score is 0.030073164741396256
	experiment 3:	silhouette score is 0.03144089126967461
	experiment 4:	silhouette score is 0.030208127555481978
	experiment 5:	silhouette score is 0.03052257512138443
	experiment 6:	silhouette score is 0.03137842279931367
	experiment 7:	silhouette score is 0.029924382868069307
	experiment 8:	silhouette score is 0.02954450746237369
	experiment 9:	silhouette score is 0.028817137118555904
	
	experiment 11: silhouette score is 0.03208045590965395
	experiment 12: silhouette score is 0.030714486373260515
	experiment 13: silhouette score is 0.03235828537526648
```

​	score reported to ***eval/FPS_2000_ss.log***

​	FPS sample index saved to ***eval/subsample/FPS/***  

2. Nearest Neighbor sampling 2000 instances and then calculate silhouette score 

   [Implementation](https://github.com/YefanZhou/ECCV2020_rebuttal/blob/759377ef500955409bf8237d4782df0ef700fa90/eval/subsample_selection_yefan.py#L207)

   **To run the script:** 

```
cd eval
```

modify ***subsample_selection_yefan.py***  line 298 as 

```python
subsample_type = "NPS"
```

```bash
python subsample_selection_yefan.py
Output:
	experiment 0: silhouette score is 0.12091750957806079
	experiment 1: silhouette score is 0.12097931233345019
	experiment 2: silhouette score is 0.1451042807407922
	experiment 3: silhouette score is 0.12605981513071832
	experiment 4: silhouette score is 0.14594201979484606
	experiment 5: silhouette score is 0.12109978570945026
	experiment 6: silhouette score is 0.12079178273729868
	experiment 7: silhouette score is 0.13148367625058213
	experiment 8: silhouette score is 0.11866163314265958
	experiment 9: silhouette score is 0.1209330393796381
```

​	score reported to ***eval/NPS_2000_ss.log***

​	NPS sample index saved to ***eval/subsample/NPS/***  

##### **May 28th**

1. sample GT/Clustering method/Autoprediction from 10432 to 2000 

   the sample index in saved to ***eval/subsample/downsample_all***   file ***name random_2000_index_exp_{}.npy*** index from 0 ~ 9

   results : SScore are saved to [table](https://docs.google.com/spreadsheets/d/1f9TXRBMoYlkFO8grtUaFqnt_FG3xRHnXsuo9op5dk30/edit#gid=0) 

   implementation: [pipline](https://github.com/YefanZhou/ECCV2020_rebuttal/blob/3ba1f0e2ade992a03928a03e293f0bb1ab4818b1/eval/compute_prediction.py#L101)

```bash
cd eval
python compute_prediction.py
```

2. modify NPS in May 27th,  use FPS to sample 500 cluster, and sample 4 nearest neighbor around the cluster

   results : SScore are saved to [table](https://docs.google.com/spreadsheets/d/1f9TXRBMoYlkFO8grtUaFqnt_FG3xRHnXsuo9op5dk30/edit#gid=0) 

   implementation: [function](https://github.com/YefanZhou/ECCV2020_rebuttal/blob/3ba1f0e2ade992a03928a03e293f0bb1ab4818b1/eval/subsample_selection_yefan.py#L245)

```bash
cd eval
python subsample_selection_yefan.py
```



##### May 29th

1. modify NPS sampling number, use 400 ( 20 x 20 ) sampling as final score reported in rebuttal

   the sampling index is saved to 

   ***subsample/NPS_500/NPS_500_samplenum_400_index_{}.npy***

   ***subsample/FPS/FPS_samplenum_400_index_{}.npy***

   ***subsample/random/random_samplenum_400_index_{}.npy***

2. **Main paper F-score stats:** 

   Oracle_NN [0.008673072329921813, 0.05986568782299093, 0.2856820629425274, 0.8010470728484635, 0.9646949746034704, 0.9975754100650117]
   resnet_twofold [0.007929509132887485, 0.054028235302803464, 0.2593303081736749, 0.7750192063911971, 0.954081717118564, 0.9953836148895361]
   label5_ae [0.007988000499098601, 0.056429490202156755, 0.28739115999791937, 0.8084866583164294, 0.9613784544891562, 0.9958016203036085]

|       F-score        |  0.5%   |  1.0%   |  2.0%   |  5.0%   |  10.0%  |  20.0%  |
| :------------------: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
|      Oracle-NN       | 0.00867 | 0.05986 | 0.28568 | 0.80104 | 0.96469 | 0.99757 |
| AE(resnet_two_fold)  | 0.00792 | 0.05402 | 0.25933 | 0.77501 | 0.95408 | 0.99538 |
|    AE(label5_ae)     | 0.00798 | 0.05642 | 0.28739 | 0.80848 | 0.96137 | 0.99580 |
| **Chamfer Distance** |         |         |         |         |         |         |
|      Oracle-NN       | 0.0719  |         |         |         |         |         |
| AE(resnet_two_fold)  | 0.0768  |         |         |         |         |         |
|    AE(label5_ae)     | 0.0716  |         |         |         |         |         |

