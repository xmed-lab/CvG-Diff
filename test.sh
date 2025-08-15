dataset_shape=256  # CT image size (squared)
res_dir='/nfs/usrhome/jchenhu/CvG-Diff/logs' # enter your directory that stores trained models
dataset_path='/home/jchenhu/dataset/aapm16/test_img' # enter your dataset path of test images
network='CvG-Diff'

# 18-view

CUDA_VISIBLE_DEVICES="0" python colddiff_main.py \
--num_views 18 \
--dataset_name 'aapm' --dataset_shape $dataset_shape \
--network $network \
--trainer_mode 'test' \
--split 'test' \
--dataset_path $dataset_path \
--net_checkpath $res_dir'/cvgdiff/ckpt/CvG-Diff/CvG-Diff-net-colddiff_best_epoch.pkl' \
--tester_save_path $res_dir \
--tester_save_name $network'/test_18v' \
--tester_save_image \
--budget_ratio 2 --time_back_ssim_threshold 0.97 --refine_budget 4

# 36-view

CUDA_VISIBLE_DEVICES="0" python colddiff_main.py \
--num_views 36 \
--dataset_name 'aapm' --dataset_shape $dataset_shape \
--network $network \
--trainer_mode 'test' \
--split 'test' \
--dataset_path $dataset_path \
--net_checkpath $res_dir'/cvgdiff/ckpt/CvG-Diff/CvG-Diff-net-colddiff_best_epoch.pkl' \
--tester_save_path $res_dir \
--tester_save_name $network'/test_36v' \
--tester_save_image \
--budget_ratio 2 --time_back_ssim_threshold 0.97 --refine_budget 4

# 72-view

CUDA_VISIBLE_DEVICES="0" python colddiff_main.py \
--num_views 72 \
--dataset_name 'aapm' --dataset_shape $dataset_shape \
--network $network \
--trainer_mode 'test' \
--split 'test' \
--dataset_path $dataset_path \
--net_checkpath $res_dir'/cvgdiff/ckpt/CvG-Diff/CvG-Diff-net-colddiff_best_epoch.pkl' \
--tester_save_path $res_dir \
--tester_save_name $network'/test_72v' \
--tester_save_image \
--budget_ratio 1 --time_back_ssim_threshold 0.97 --refine_budget 4