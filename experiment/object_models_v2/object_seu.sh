# Mutual Parameter
MODE="object"                          # dataset mode 
SAMPLERATIO=1.0                        # ratio to sample the dataset
EPOCH=70                               # total epoch for training 
TRAINBATCH=64                          # training batch size 
TESTBATCH=200                          # test batch size 
VALBATCH=200                           # val batch size 
VIEWS="01234"                          # views
LR=1e-3                                # learning rate
MOMENTUM=0.9                           # momentum 
DECAYSTEP=50                           # learning decay step 
WEIGHTDECAY=1e-6                       # weight decay
BNSTATS_STEP=20                        # step to do batch norm stats
TEST_STEP=10                           # step to do test
NUM_WORKER=4                 
# CUDA 
CUDA1=0                                # gpu id for model1  


# Model 1 FoldingNet with 6 layer fc decoder two-folding(loss backward)  (submission for ECCV2020)
CUDA_VISIBLE_DEVICES=$CUDA1 python ../../train.py \
                          --data-basedir '../../../What3D' \
						  --ptcloud-path "ptcloud_n.npz" \
             		      --model "foldingres"      `# model Info`\
                          --SVR \
                          --num_layers 6 \
                          --hidden_neurons 512 \
                          --bottleneck_size 512 \
                          --remove_all_batchNorms \
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
						  --log-dir	'foldingres6/'  `# logging Info`\
						  --tensorboard \
						  --save-results \
						  --test \
						  --train 