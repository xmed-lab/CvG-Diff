epochs=40  # Number of epochs
dataset_shape=256  # CT image size (squared)
res_dir='/nfs/usrhome/jchenhu/CvG-Diff/logs'  # enter your directory for storing result
dataset_path='/home/jchenhu/dataset/aapm16/train_img' # enter your dataset path of train images
network='CvG-Diff'

CUDA_VISIBLE_DEVICES="0" python colddiff_main.py --epochs $epochs \
--lr 4e-5 --optimizer 'adam' \
--dataset_name 'aapm' --dataset_shape $dataset_shape \
--network $network \
--loss 'l2' --trainer_mode 'train' \
--checkpoint_root $res_dir'/cvgdiff/ckpt' \
--checkpoint_dir $network \
--dataset_path $dataset_path \
--batch_size 4 --num_workers 4 --log_interval 200 \
--use_tqdm \
--use_wandb --run_name 'CvG-Diff' \
--wandb_root $res_dir'/cvgdiff/wandb' \
--wandb_dir $network