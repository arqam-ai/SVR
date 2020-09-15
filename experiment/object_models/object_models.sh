# Mutual Parameter
SAMPLERATIO=0.01                        # ratio to sample the dataset
EPOCH=10                                # total epoch for training 
TRAINBATCH=64                          # training batch size 
TESTBATCH=200                          # test batch size 
VALBATCH=200                           # val batch size 
MODE="object"                          # dataset mode 
VIEWS="01234"                          # views
LR=3e-3                                # learning rate
MOMENTUM=0.9                           # momentum 
DECAYSTEP=3                            # learning decay step 
WEIGHTDECAY=1e-6                       # weight decay
'''
python ../../train.py --data-basedir '../../../What3D' \
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
						  --output-dir 'object_foldingres6/' \
						  --snapshot-dir 'object_foldingres6/' \
						  --log-dir	'object_foldingres6/'	\
						  --lr-G $LR \
						  --momentum $MOMENTUM \
						  --weight-decay $WEIGHTDECAY \
						  --lr_decay_step $DECAYSTEP \
						  --train \
						  --test \
						  --tensorboard \
						  --folding-twice \
						  --if-BNstats \
						  --save-results 
'''
python ../../train.py --data-basedir '../../../What3D' \
						  --ptcloud-path "ptcloud_0.npz" \
						  --model "foldingres18" \
						  --image-size 224 \
						  --view $VIEWS \
						  --sample-ratio $SAMPLERATIO \
						  --total-epoch $EPOCH \
						  --train-batch-size $TRAINBATCH \
						  --test-batch-size $TESTBATCH \
						  --val-batch-size $VALBATCH \
						  --verbose_per_n_batch 10 \
						  --test_per_n_epoch 5 \
						  --lambda-loss-fine 1. \
						  --lambda-loss-primitive 1. \
						  --lambda-loss-classification 0. \
						  --mode $MODE \
						  --output-dir 'object_foldingres18/' \
						  --snapshot-dir 'object_foldingres18/' \
						  --log-dir	'object_foldingres18/'	\
						  --lr-G $LR \
						  --momentum $MOMENTUM \
						  --weight-decay 1e-6 \
						  --lr_decay_step $DECAYSTEP \
						  --train \
						  --test \
						  --tensorboard \
						  --folding-twice \
						  --if-BNstats \
						  --save-results

'''
python ../../train.py --data-basedir '../../../What3D' \
						  --ptcloud-path "ptcloud_0.npz" \
						  --model "psgn" \
						  --image-size 224 \
						  --view $VIEWS \
						  --sample-ratio $SAMPLERATIO \
						  --total-epoch $EPOCH \
						  --train-batch-size $TRAINBATCH \
						  --test-batch-size $TESTBATCH \
						  --val-batch-size $VALBATCH \
						  --verbose_per_n_batch 10 \
						  --test_per_n_epoch 5 \
						  --lambda-loss-fine 1. \
						  --lambda-loss-primitive 1. \
						  --lambda-loss-classification 0. \
						  --mode $MODE \
						  --output-dir 'object_psgn/' \
						  --snapshot-dir 'object_psgn/' \
						  --log-dir	'object_psgn/'	\
						  --lr-G $LR \
						  --momentum $MOMENTUM \
						  --weight-decay 1e-6 \
						  --lr_decay_step $DECAYSTEP \
						  --train \
						  --test \
						  --tensorboard \
						  --save-results
'''						  
