# Mutual Parameter
MODE="viewer"                          # dataset mode 
SAMPLERATIO=1.0                        # ratio to sample the dataset
EPOCH=70                               # total epoch for training 
TRAINBATCH=64                          # training batch size 
TESTBATCH=64                          # test batch size 
VALBATCH=64                           # val batch size 
VIEWS="01234"                          # views
LR=1e-3                                # learning rate
MOMENTUM=0.9                           # momentum 
DECAYSTEP=30                           # learning decay step 
WEIGHTDECAY=1e-6                       # weight decay
BNSTATS_STEP=20                        # step to do batch norm stats
TEST_STEP=10                           # step to do test
NUM_WORKER=4                 
# CUDA 
CUDA1=3                                # gpu id for model1  foldingres no BN hidden 6 
CUDA2=0                                # gpu id for model1  foldingres BN hidden 6 
CUDA3=0                                # gpu id for model1  batch size = 32 foldingres BN hidden 6


CUDA_VISIBLE_DEVICES=$CUDA1 python ../../train.py \
                          --data-basedir '../../../What3D' \
						  --ptcloud-path "ptcloud_n.npz" \
             		      --model "atlasnet"      `# model Info`\
                          --SVR \
                          --num_layers 6 \
                          --hidden_neurons 512 \
                          --bottleneck_size 512 \
                          `# --remove_all_batchNorms`\
						  --nb_primitives 4       `# atlasnet Info 1 4 16`\
 						  --template_type "SQUARE" \
                          --mode $MODE             `# dataset Info`\
						  --image-size 224 \
						  --view $VIEWS \
						  --sample-ratio $SAMPLERATIO \
                          --num-worker $NUM_WORKER \
						  --total-epoch $EPOCH      `# training Info`\
						  --train-batch-size $TRAINBATCH \
						  --test-batch-size $TESTBATCH \
						  --val-batch-size $VALBATCH \
						  --verbose_per_n_batch 50 \
						  --test_per_n_epoch $TEST_STEP \
						  --lambda-loss-fine 1. \
						  --lambda-loss-primitive 1. \
                          --lr-G $LR \
                          --momentum $MOMENTUM \
                          --lr_decay_step $DECAYSTEP \
                          --weight-decay $WEIGHTDECAY \
						  --log-dir	'atlasnet_viewer_4square_lr6_random_ucb/'  `# logging Info`\
						  --tensorboard \
						  --save-results \
						  --test \
						  --train 

# # model2  foldingres BN hidden 6 batch size = 64
# CUDA_VISIBLE_DEVICES=$CUDA2 python ../../train.py \
#                           --data-basedir '../../../What3D' \
# 						  --ptcloud-path "ptcloud_n.npz" \
#              		      --model "foldingres"      `# model Info`\
#                           --SVR \
#                           --num_layers 6 \
#                           --hidden_neurons 512 \
#                           --bottleneck_size 512 \
# 						  `# --remove_all_batchNorms`\
#                           --mode $MODE             `# dataset Info`\
# 						  --image-size 224 \
# 						  --view $VIEWS \
# 						  --sample-ratio $SAMPLERATIO \
#                           --num-worker $NUM_WORKER \
# 						  --total-epoch $EPOCH      `# training Info`\
# 						  --train-batch-size $TRAINBATCH \
# 						  --test-batch-size $TESTBATCH \
# 						  --val-batch-size $VALBATCH \
# 						  --verbose_per_n_batch 50 \
# 						  --test_per_n_epoch $TEST_STEP \
# 						  --lambda-loss-fine 1. \
# 						  --lambda-loss-primitive 1. \
#                           --lr-G $LR \
#                           --momentum $MOMENTUM \
#                           --lr_decay_step $DECAYSTEP \
#                           --weight-decay $WEIGHTDECAY \
# 						  --log-dir	'foldingres6_FCBN/'  `# logging Info`\
# 						  --tensorboard \
# 						  --save-results \
# 						  --test \
# 						  --train &

# # model3  foldingres BN hidden 6 batch size = 32
# CUDA_VISIBLE_DEVICES=$CUDA3 python ../../train.py \
#                           --data-basedir '../../../What3D' \
# 						  --ptcloud-path "ptcloud_n.npz" \
#              		      --model "foldingres"      `# model Info`\
#                           --SVR \
#                           --num_layers 6 \
#                           --hidden_neurons 512 \
#                           --bottleneck_size 512 \
# 						  `# --remove_all_batchNorms`\
#                           --mode $MODE             `# dataset Info`\
# 						  --image-size 224 \
# 						  --view $VIEWS \
# 						  --sample-ratio $SAMPLERATIO \
#                           --num-worker $NUM_WORKER \
# 						  --total-epoch $EPOCH      `# training Info`\
# 						  `#--train-batch-size $TRAINBATCH`\
# 						  --train-batch-size 32 \
# 						  --test-batch-size $TESTBATCH \
# 						  --val-batch-size $VALBATCH \
# 						  --verbose_per_n_batch 50 \
# 						  --test_per_n_epoch $TEST_STEP \
# 						  --lambda-loss-fine 1. \
# 						  --lambda-loss-primitive 1. \
#                           --lr-G $LR \
#                           --momentum $MOMENTUM \
#                           --lr_decay_step $DECAYSTEP \
#                           --weight-decay $WEIGHTDECAY \
# 						  --log-dir	'foldingres6_FCBN_BS32/'  `# logging Info`\
# 						  --tensorboard \
# 						  --save-results \
# 						  --test \
# 						  --train &