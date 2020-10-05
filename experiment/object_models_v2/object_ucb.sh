# Mutual Parameter
MODE="object"                          # dataset mode 
SAMPLERATIO=1.0                        # ratio to sample the dataset
EPOCH=70                               # total epoch for training 
TRAINBATCH=64                          # training batch size 
TESTBATCH=64                           # test batch size 
VALBATCH=64                           # val batch size 
VIEWS="01234"                          # views
LR=1e-3                                # learning rate
MOMENTUM=0.9                           # momentum 
DECAYSTEP=30                           # learning decay step 
WEIGHTDECAY=1e-6                       # weight decay
BNSTATS_STEP=20                        # step to do batch norm stats
TEST_STEP=10                           # step to do test
NUM_WORKER=4                 
# CUDA 4
CUDA1=0                                # gpu id for model  atlasnet lr=2  4 square
CUDA2=1                                # gpu id for model  atlasnet lr=6  4 square
CUDA3=2                                # gpu id for model  atlasnet 1r=2  8 square
CUDA4=3                                # gpu id for model  atlasnet 1r=2  16 square
CUDA5=1                                # gpu id for model  foldingres BN hidden 6 
CUDA6=1                                # gpu id for model  batch size = 32 foldingres BN hidden 6


# model4 atlasnet BN hidden 6 batch size = 64
# CUDA_VISIBLE_DEVICES=$CUDA1 python ../../train.py \
#                           --data-basedir '../../../What3D' \
# 						  --ptcloud-path "ptcloud_n.npz" \
#              		      --model "atlasnet"      `# model Info`\
#                           --SVR \
#                           --num_layers 2 \
#                           --hidden_neurons 512 \
#                           --bottleneck_size 512 \
#                           `# --remove_all_batchNorms`\
# 						  --nb_primitives 4       `# atlasnet Info 1 4 16`\
#  						  --template_type "SQUARE" \
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
# 						  --log-dir	'atlasnet_object_4square_lr2_random_ucb/'  `# logging Info`\
# 						  --tensorboard \
# 						  --save-results \
# 						  --test \
# 						  --train 

# # model4 atlasnet BN hidden 6 batch size = 64
# CUDA_VISIBLE_DEVICES=$CUDA2 python ../../train.py \
#                           --data-basedir '../../../What3D' \
# 						  --ptcloud-path "ptcloud_n.npz" \
#              		      --model "atlasnet"      `# model Info`\
#                           --SVR \
#                           --num_layers 6 \
#                           --hidden_neurons 512 \
#                           --bottleneck_size 512 \
#                           `# --remove_all_batchNorms`\
# 						  --nb_primitives 4       `# atlasnet Info 1 4 16`\
#  						  --template_type "SQUARE" \
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
# 						  --log-dir	'atlasnet_object_4square_lr6_random_ucb/'  `# logging Info`\
# 						  --tensorboard \
# 						  --save-results \
# 						  --test \
# 						  --train &


# # model4 atlasnet BN hidden 6 batch size = 64
# CUDA_VISIBLE_DEVICES=$CUDA3 python ../../train.py \
#                           --data-basedir '../../../What3D' \
# 						  --ptcloud-path "ptcloud_n.npz" \
#              		      --model "atlasnet"      `# model Info`\
#                           --SVR \
#                           --num_layers 2 \
#                           --hidden_neurons 512 \
#                           --bottleneck_size 512 \
#                           `# --remove_all_batchNorms`\
# 						  --nb_primitives 8       `# atlasnet Info 1 4 16`\
#  						  --template_type "SQUARE" \
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
# 						  --log-dir	'atlasnet_object_8square_lr2_random_ucb/'  `# logging Info`\
# 						  --tensorboard \
# 						  --save-results \
# 						  --test \
# 						  --train &

# model4 atlasnet BN hidden 6 batch size = 64
CUDA_VISIBLE_DEVICES=$CUDA4 python ../../train.py \
                          --data-basedir '../../../What3D' \
						  --ptcloud-path "ptcloud_n.npz" \
             		      --model "atlasnet"      `# model Info`\
                          --SVR \
                          --num_layers 2 \
                          --hidden_neurons 512 \
                          --bottleneck_size 512 \
                          `# --remove_all_batchNorms`\
						  --nb_primitives 16       `# atlasnet Info 1 4 16`\
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
						  --log-dir	'atlasnet_object_16square_lr2_random_ucb/'  `# logging Info`\
						  --tensorboard \
						  --save-results \
						  --test \
						  --train