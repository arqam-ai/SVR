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
# CUDA 
CUDA1=0                                # gpu id for model  atlasnet BS 64
CUDA2=1                                # gpu id for model  foldingres BN hidden 6 
CUDA3=1                                # gpu id for model  batch size = 32 foldingres BN hidden 6
CUDA4=0                                # gpu id for model  atlasnet 1 square
CUDA5=1                                # gpu id for model  foldingres BN hidden 6 
CUDA6=1                                # gpu id for model  batch size = 32 foldingres BN hidden 6


# model4 atlasnet BN hidden 6 batch size = 64
CUDA_VISIBLE_DEVICES=$CUDA1 python ../../train.py \
                          --data-basedir '../../../What3D' \
						  --ptcloud-path "ptcloud_n.npz" \
             		      --model "atlasnet"      `# model Info`\
                          --SVR \
                          --num_layers 2 \
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
						  --log-dir	'atlasnet_object_4square_lr2_random/'  `# logging Info`\
						  --tensorboard \
						  --save-results \
						  --test \
						  --train 

# # # model foldingres batch size = 32 width = 1024
# CUDA_VISIBLE_DEVICES=$CUDA5 python ../../train.py \
#                           --data-basedir '../../../What3D' \
# 						  --ptcloud-path "ptcloud_n.npz" \
#              		      --model "foldingres"      `# model Info`\
#                           --SVR \
#                           --num_layers 6 \
#                           --hidden_neurons 1024 \
#                           --bottleneck_size 1024 \
#                           `# --remove_all_batchNorms`\
#                           --mode $MODE             `# dataset Info`\
# 						  --image-size 224 \
# 						  --view $VIEWS \
# 						  --sample-ratio $SAMPLERATIO \
#                           --num-worker $NUM_WORKER \
# 						  --total-epoch $EPOCH      `# training Info`\
# 						  `#--train-batch-size $TRAINBATCH` \
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
# 						  --log-dir	'foldingres6_FCBN_BS32_W1024/'  `# logging Info`\
# 						  --tensorboard \
# 						  --save-results \
# 						  --test &
# #						  --train 


# # model atlasnet BN hidden 6 batch size = 32
# CUDA_VISIBLE_DEVICES=$CUDA6 python ../../train.py \
#                           --data-basedir '../../../What3D' \
# 						  --ptcloud-path "ptcloud_n.npz" \
#              		      --model "atlasnet"      `# model Info`\
#                           --SVR \
#                           --num_layers 6 \
#                           --hidden_neurons 1024 \
#                           --bottleneck_size 1024 \
#                           `# --remove_all_batchNorms`\
# 						  --nb_primitives 1       `# atlasnet Info 1 4 16`\
#  						  --template_type "SQUARE" \
#                           --mode $MODE             `# dataset Info`\
# 						  --image-size 224 \
# 						  --view $VIEWS \
# 						  --sample-ratio $SAMPLERATIO \
#                           --num-worker $NUM_WORKER \
# 						  --total-epoch $EPOCH      `# training Info`\
# 						  `#--train-batch-size $TRAINBATCH` \
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
# 						  --log-dir	'atlasnet_object_1square_BS32_W1024/'  `# logging Info`\
# 						  --tensorboard \
# 						  --save-results \
# 						  --test &
# #						  --train &


# model4 atlasnet BN hidden 6 batch size = 64
# CUDA_VISIBLE_DEVICES=$CUDA4 python ../../train.py \
#                           --data-basedir '../../../What3D' \
# 						  --ptcloud-path "ptcloud_n.npz" \
#              		      --model "atlasnet"      `# model Info`\
#                           --SVR \
#                           --num_layers 6 \
#                           --hidden_neurons 512 \
#                           --bottleneck_size 512 \
#                           `# --remove_all_batchNorms`\
# 						  --nb_primitives 1       `# atlasnet Info 1 4 16`\
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
# 						  --log-dir	'atlasnet_object_1square/'  `# logging Info`\
# 						  --tensorboard \
# 						  --save-results \
# 						  --test \
# 						  --train &
