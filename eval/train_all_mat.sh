#!/bin/bash
declare -i TRENDS=17                       # Trends used to compute 
declare -i MAT_SIZE=36757                  # Size of distance matrix
declare -i START_INDEX=0 
declare -i STEP=100                        # Step between START_INDEX and END_INDEX  
declare -i END_INDEX=0
END_INDEX=`expr $START_INDEX + $STEP`
declare -i INCREMENTAL_STEP=300

for i in $(seq 1 1 $TRENDS) 
do  
    if (( i < 9 )); then
        python -u trainset_dis_mat.py --ptcloud-path "/home/zyf/What3D/ptcloud_0.npz" \
                           --matrix-save-path "/home/../public/zyf/ECCV2020/distance_matrix" \
                           --experiment-name "PC_CD_partial" \
                           --gpu_id "0" \
                           --distance_matrix "CD" \
                           --start_index $START_INDEX \
                           --end_index $END_INDEX &
        
    else
        python -u trainset_dis_mat.py --ptcloud-path "/home/zyf/What3D/ptcloud_0.npz" \
                           --matrix-save-path "/home/../public/zyf/ECCV2020/distance_matrix" \
                           --experiment-name "PC_CD_partial" \
                           --gpu_id "1" \
                           --distance_matrix "CD" \
                           --start_index $START_INDEX \
                           --end_index $END_INDEX &
    fi
    START_INDEX=$END_INDEX
    END_INDEX=`expr $END_INDEX + $STEP`
    STEP=`expr $STEP + $INCREMENTAL_STEP`
    if (( END_INDEX > MAT_SIZE )); then 
        END_INDEX=$MAT_SIZE
    fi
    if (( START_INDEX == MAT_SIZE )); then
        break
    fi
done
wait
python convert_block_to_complete_distance_matrix.py --partial_path '/home/../public/zyf/ECCV2020/distance_matrix/PC_CD_partial' \
                                                    --save_path '/home/../public/zyf/ECCV2020/distance_matrix/GT_Trainset_dis_matrix.npy' \
                                                    --mat_size $MAT_SIZE
