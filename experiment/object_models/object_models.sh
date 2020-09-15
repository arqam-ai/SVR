# Mutual Parameter
SAMPLERATIO=1.0                        # ratio to sample the dataset
EPOCH=100                                # total epoch for training 
TRAINBATCH=64                          # training batch size 
TESTBATCH=200                          # test batch size 
VALBATCH=200                           # val batch size 
MODE="object"                          # dataset mode 
VIEWS="01234"                          # views
LR=3e-3                                # learning rate
MOMENTUM=0.9                           # momentum 
DECAYSTEP=40                           # learning decay step 
WEIGHTDECAY=1e-6                       # weight decay
BNSTATS_STEP=30                        # step to do batch norm stats
# CUDA 
CUDA1=0                                # gpu id for model1 
CUDA2=0                                # gpu id for model2 
CUDA3=1                                # gpu id for model3 
CUDA4=1                                # gpu id for model4 

## FoldingNet with 6 layer fc decoder two-folding(loss backward)  (submission for ECCV2020)
CUDA_VISIBLE_DEVICES=$CUDA1 python ../../train.py --data-basedir '../../../What3D' \
						  --ptcloud-path "ptcloud_0.npz" \
						  --model "foldingres6" \
						  --image-size 224 \
						  --view $VIEWS \
						  --sample-ratio $SAMPLERATIO \
						  --total-epoch $EPOCH \
						  --train-batch-size $TRAINBATCH \
						  --test-batch-size $TESTBATCH \
						  --val-batch-size $VALBATCH \
						  --verbose_per_n_batch 20 \
						  --test_per_n_epoch 5 \
						  --lambda-loss-fine 1. \
						  --lambda-loss-primitive 1. \
						  --lambda-loss-classification 0. \
						  --mode $MODE \
						  --output-dir 'foldingres6_twofold/' \
						  --snapshot-dir 'foldingres6_twofold/' \
						  --log-dir	'foldingres6_twofold/'	\
						  --lr-G $LR \
						  --momentum $MOMENTUM \
						  --weight-decay $WEIGHTDECAY \
						  --lr_decay_step $DECAYSTEP \
						  --train \
						  --test \
						  --tensorboard \
						  --folding-twice \
						  --if-BNstats \
						  --bnstats-step $BNSTATS_STEP \
						  --save-results &

## FoldingNet with 6 layer fc decoder only ONE-folding(loss backward)  
CUDA_VISIBLE_DEVICES=$CUDA2 python ../../train.py --data-basedir '../../../What3D' \
						  --ptcloud-path "ptcloud_0.npz" \
						  --model "foldingres6" \
						  --image-size 224 \
						  --view $VIEWS \
						  --sample-ratio $SAMPLERATIO \
						  --total-epoch $EPOCH \
						  --train-batch-size $TRAINBATCH \
						  --test-batch-size $TESTBATCH \
						  --val-batch-size $VALBATCH \
						  --verbose_per_n_batch 20 \
						  --test_per_n_epoch 5 \
						  --lambda-loss-fine 1. \
						  --lambda-loss-primitive 1. \
						  --lambda-loss-classification 0. \
						  --mode $MODE \
						  --output-dir 'foldingres6/' \
						  --snapshot-dir 'foldingres6/' \
						  --log-dir	'foldingres6/'	\
						  --lr-G $LR \
						  --momentum $MOMENTUM \
						  --weight-decay $WEIGHTDECAY \
						  --lr_decay_step $DECAYSTEP \
						  --train \
						  --test \
						  --tensorboard \
						  --if-BNstats \
						  --bnstats-step $BNSTATS_STEP \
						  --save-results &

## FoldingNet with 18 layer fc decoder only ONE-folding  
CUDA_VISIBLE_DEVICES=$CUDA3 python ../../train.py --data-basedir '../../../What3D' \
						  --ptcloud-path "ptcloud_0.npz" \
						  --model "foldingres18" \
						  --image-size 224 \
						  --view $VIEWS \
						  --sample-ratio $SAMPLERATIO \
						  --total-epoch $EPOCH \
						  --train-batch-size $TRAINBATCH \
						  --test-batch-size $TESTBATCH \
						  --val-batch-size $VALBATCH \
						  --verbose_per_n_batch 20 \
						  --test_per_n_epoch 5 \
						  --lambda-loss-fine 1. \
						  --lambda-loss-primitive 1. \
						  --lambda-loss-classification 0. \
						  --mode $MODE \
						  --output-dir 'foldingres18/' \
						  --snapshot-dir 'foldingres18/' \
						  --log-dir	'foldingres18/'	\
						  --lr-G $LR \
						  --momentum $MOMENTUM \
						  --weight-decay $WEIGHTDECAY \
						  --lr_decay_step $DECAYSTEP \
						  --train \
						  --test \
						  --tensorboard \
						  --if-BNstats \
						  --bnstats-step $BNSTATS_STEP \
						  --save-results &

## PSGN Haoqiang Fan 
CUDA_VISIBLE_DEVICES=$CUDA4 python ../../train.py --data-basedir '../../../What3D' \
						  --ptcloud-path "ptcloud_0.npz" \
						  --model "psgn" \
						  --image-size 224 \
						  --view $VIEWS \
						  --sample-ratio $SAMPLERATIO \
						  --total-epoch $EPOCH \
						  --train-batch-size $TRAINBATCH \
						  --test-batch-size $TESTBATCH \
						  --val-batch-size $VALBATCH \
						  --verbose_per_n_batch 20 \
						  --test_per_n_epoch 5 \
						  --lambda-loss-fine 1. \
						  --lambda-loss-primitive 1. \
						  --lambda-loss-classification 0. \
						  --mode $MODE \
						  --output-dir 'psgn/' \
						  --snapshot-dir 'psgn/' \
						  --log-dir	'psgn/'	\
						  --lr-G $LR \
						  --momentum $MOMENTUM \
						  --weight-decay $WEIGHTDECAY \
						  --lr_decay_step $DECAYSTEP \
						  --train \
						  --test \
						  --tensorboard \
						  --save-results &

