
'''
nohup python ../../train.py --data-basedir '/home/zyf/What3D' \
						  --ptcloud-path "ptcloud_0.npz" \
						  --model "foldingnet" \
						  --image-size 224 \
						  --view "01234" \
						  --sample-ratio 1.0 \
						  --total-epoch 20 \
						  --train-batch-size 64 \
						  --test-batch-size 200 \
						  --val-batch-size 200 \
						  --verbose_per_n_batch 25 \
						  --test_per_n_epoch 1 \
						  --lr-G 3e-3 \
						  --lambda-loss-fine 1. \
						  --lambda-loss-primitive 1. \
						  --lambda-loss-classification 0. \
						  --output-dir 'viewer_foldingnet/' \
						  --snapshot-dir 'viewer_foldingnet/' \
						  --log-dir	'viewer_foldingnet/'	\
						  --momentum 0.9 \
						  --weight-decay 1e-6 \
						  --lr_decay_step 7 \
						  --train \
						  --test \
						  --tensorboard \
						  --folding-twice &
#						  --save-results \
'''

'''
nohup python ../../train.py --data-basedir '/home/zyf/What3D' \
						  --ptcloud-path "ptcloud_0.npz" \
						  --model "psgn" \
						  --image-size 224 \
						  --view "01234" \
						  --sample-ratio 1.0 \
						  --total-epoch 20 \
						  --train-batch-size 64 \
						  --test-batch-size 200 \
						  --val-batch-size 200 \
						  --verbose_per_n_batch 25 \
						  --test_per_n_epoch 1 \
						  --lr-G 3e-3 \
						  --lambda-loss-fine 1. \
						  --lambda-loss-primitive 1. \
						  --lambda-loss-classification 0. \
						  --output-dir 'viewer_psgn/' \
						  --snapshot-dir 'viewer_psgn/' \
						  --log-dir	'viewer_psgn/'	\
						  --momentum 0.9 \
						  --weight-decay 1e-6 \
						  --lr_decay_step 7 \
						  --train \
						  --test \
						  --tensorboard \
						  --folding-twice &
'''

nohup python ../../train.py --data-basedir '/home/zyf/What3D' \
						  --ptcloud-path "ptcloud_0.npz" \
						  --model "foldingnet" \
						  --image-size 224 \
						  --view "01234" \
						  --sample-ratio 1.0 \
						  --total-epoch 20 \
						  --train-batch-size 64 \
						  --test-batch-size 200 \
						  --val-batch-size 200 \
						  --verbose_per_n_batch 25 \
						  --test_per_n_epoch 1 \
						  --lr-G 3e-3 \
						  --lambda-loss-fine 1. \
						  --lambda-loss-primitive 1. \
						  --lambda-loss-classification 0. \
						  --output-dir 'viewer_foldingone/' \
						  --snapshot-dir 'viewer_foldingone/' \
						  --log-dir	'viewer_foldingone/'	\
						  --momentum 0.9 \
						  --weight-decay 1e-6 \
						  --lr_decay_step 7 \
						  --train \
						  --test \
						  --tensorboard &