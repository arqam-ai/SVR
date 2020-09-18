# Mutual Parameter
SAMPLERATIO=1.0                        # ratio to sample the dataset
EPOCH=70                               # total epoch for training 
TRAINBATCH=64                           # training batch size 
TESTBATCH=200                          # test batch size 
VALBATCH=200                           # val batch size 
MODE="object"                          # dataset mode 
VIEWS="01234"                          # views
LR=3e-3                                # learning rate
MOMENTUM=0.9                           # momentum 
DECAYSTEP=30                           # learning decay step 
WEIGHTDECAY=1e-6                       # weight decay
BNSTATS_STEP=20                        # step to do batch norm stats
TEST_STEP=10                           # step to do test
# CUDA 
CUDA1=0                                # gpu id for model1 
CUDA2=0                                # gpu id for model2 
CUDA3=1                                # gpu id for model3 
CUDA4=1                                # gpu id for model4 
CUDA5=1                                # gpu id for model5

## FoldingNet with 6 layer fc decoder two-folding(loss backward)  (submission for ECCV2020)
CUDA_VISIBLE_DEVICES=$CUDA1 python ../../train.py --data-basedir '../../../What3D' \
						  --ptcloud-path "ptcloud_n.npz" \
						  --model "foldingres6" \
						  --image-size 224 \
						  --view $VIEWS \
						  --sample-ratio $SAMPLERATIO \
						  --total-epoch $EPOCH \
						  --train-batch-size $TRAINBATCH \
						  --test-batch-size $TESTBATCH \
						  --val-batch-size $VALBATCH \
						  --verbose_per_n_batch 50 \
						  --test_per_n_epoch $TEST_STEP \
						  --lambda-loss-fine 1. \
						  --lambda-loss-primitive 1. \
						  --mode $MODE \
						  --output-dir 'foldingres6_twofold/' \
						  --snapshot-dir 'foldingres6_twofold/' \
						  --log-dir	'foldingres6_twofold/'	\
						  --lr-G $LR \
						  --momentum $MOMENTUM \
						  --weight-decay $WEIGHTDECAY \
						  --lr_decay_step $DECAYSTEP \
						  --tensorboard \
						  --folding-twice \
						  --if-BNstats \
						  --bnstats-step $BNSTATS_STEP \
						  --save-results \
						  --test \
						  --train &

## FoldingNet with 6 layer fc decoder only ONE-folding(loss backward)  
CUDA_VISIBLE_DEVICES=$CUDA2 python ../../train.py --data-basedir '../../../What3D' \
						  --ptcloud-path "ptcloud_n.npz" \
						  --model "foldingres6" \
						  --image-size 224 \
						  --view $VIEWS \
						  --sample-ratio $SAMPLERATIO \
						  --total-epoch $EPOCH \
						  --train-batch-size $TRAINBATCH \
						  --test-batch-size $TESTBATCH \
						  --val-batch-size $VALBATCH \
						  --verbose_per_n_batch 50 \
						  --test_per_n_epoch $TEST_STEP \
						  --lambda-loss-fine 1. \
						  --mode $MODE \
						  --output-dir 'foldingres6/' \
						  --snapshot-dir 'foldingres6/' \
						  --log-dir	'foldingres6/'	\
						  --lr-G $LR \
						  --momentum $MOMENTUM \
						  --weight-decay $WEIGHTDECAY \
						  --lr_decay_step $DECAYSTEP \
						  --tensorboard \
						  --if-BNstats \
						  --bnstats-step $BNSTATS_STEP \
						  --save-results \
						  --test \
						  --train &

## FoldingNet with 18 layer fc decoder only ONE-folding  
CUDA_VISIBLE_DEVICES=$CUDA3 python ../../train.py --data-basedir '../../../What3D' \
						  --ptcloud-path "ptcloud_n.npz" \
						  --model "foldingres18" \
						  --image-size 224 \
						  --view $VIEWS \
						  --sample-ratio $SAMPLERATIO \
						  --total-epoch $EPOCH \
						  --train-batch-size $TRAINBATCH \
						  --test-batch-size $TESTBATCH \
						  --val-batch-size $VALBATCH \
						  --verbose_per_n_batch 50 \
						  --test_per_n_epoch $TEST_STEP \
						  --lambda-loss-fine 1. \
						  --mode $MODE \
						  --output-dir 'foldingres18/' \
						  --snapshot-dir 'foldingres18/' \
						  --log-dir	'foldingres18/'	\
						  --lr-G $LR \
						  --momentum $MOMENTUM \
						  --weight-decay $WEIGHTDECAY \
						  --lr_decay_step $DECAYSTEP \
						  --tensorboard \
						  --if-BNstats \
						  --bnstats-step $BNSTATS_STEP \
						  --save-results \
						  --test \
						  --train &

## PSGN Haoqiang Fan 
CUDA_VISIBLE_DEVICES=$CUDA4 python ../../train.py --data-basedir '../../../What3D' \
						  --ptcloud-path "ptcloud_n.npz" \
						  --model "psgn" \
						  --image-size 224 \
						  --view $VIEWS \
						  --sample-ratio $SAMPLERATIO \
						  --total-epoch $EPOCH \
						  --train-batch-size $TRAINBATCH \
						  --test-batch-size $TESTBATCH \
						  --val-batch-size $VALBATCH \
						  --verbose_per_n_batch 50 \
						  --test_per_n_epoch $TEST_STEP \
						  --lambda-loss-fine 1. \
						  --mode $MODE \
						  --output-dir 'psgn/' \
						  --snapshot-dir 'psgn/' \
						  --log-dir	'psgn/'	\
						  --lr-G $LR \
						  --momentum $MOMENTUM \
						  --weight-decay $WEIGHTDECAY \
						  --lr_decay_step $DECAYSTEP \
						  --tensorboard \
						  --save-results \
						  --test \
						  --train &

## AtlasNet 4 patches
CUDA_VISIBLE_DEVICES=$CUDA5 python ../../train.py --data-basedir '../../../What3D' \
						  --ptcloud-path "ptcloud_n.npz" \
						  --model "atlasnet" \
						  --image-size 224 \
						  --view $VIEWS \
						  --sample-ratio $SAMPLERATIO \
						  --total-epoch $EPOCH \
						  --train-batch-size $TRAINBATCH \
						  --test-batch-size $TESTBATCH \
						  --val-batch-size $VALBATCH \
						  --verbose_per_n_batch 50 \
						  --test_per_n_epoch $TEST_STEP \
						  --lambda-loss-fine 1. \
						  --mode $MODE \
						  --output-dir 'atlasnet_4patch/' \
						  --snapshot-dir 'atlasnet_4patch/' \
						  --log-dir	'atlasnet_4patch/'	\
						  --lr-G $LR \
						  --momentum $MOMENTUM \
						  --weight-decay $WEIGHTDECAY \
						  --lr_decay_step $DECAYSTEP \
						  --tensorboard \
						  --save-results \
						  --test \
						  --train &
		  

