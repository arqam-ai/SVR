# This is a code repo for ECCV rebuttal  
  
## Clustering Method to infer 3D shape (adapted from What3D)  
  
  What3D paper: https://arxiv.org/abs/1905.03678   
  What3D dataset: https://drive.google.com/drive/folders/1OmL-EFCB-I_YhvvZ_DHRpLbFYg0vYf8G  
  <br>
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

