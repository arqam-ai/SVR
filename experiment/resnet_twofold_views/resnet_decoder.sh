
nohup python ../../train.py --data-basedir '/home/../../public/zyf/What3D' \
						  --ptcloud-path "ptcloud_0.npz" \
						  --image-size 224 \
						  --view "0" \
						  --sample-ratio 1.0 \
						  --total-epoch 100 \
						  --train-batch-size 32 \
						  --test-batch-size 200 \
						  --val-batch-size 200 \
						  --verbose_per_n_batch 20 \
						  --lr-G 3e-3 \
						  --lambda-loss-fine 1. \
						  --lambda-loss-primitive 1. \
						  --lambda-loss-classification 0. \
						  --output-dir 'results_resnet_decoder/' \
						  --snapshot-dir 'snapshots_resnet_decoder/' \
						  --log-dir	'logs_resnet_decoder/'	\
						  --momentum 0.9 \
						  --weight-decay 1e-6 \
						  --lr_decay_step 40 \
						  --train \
						  --test \
						  --tensorboard \
						  --folding-twice &
#						  --save-results \