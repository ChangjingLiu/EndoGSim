# 运行仿真以生成GT数据
# cholecseg_sub/video01_00080
CUDA_VISIBLE_DEVICES=1 python simulation_gt.py \
  --model_path ./model/cholecseg_sub/video01_00080 \
  --n_epoch 50 \
  --output_path ./model/cholecseg_sub/video01_00080 \
  --physics_config ./config/cholecseg_sub/video01_00080_config_gt.json \
  --n_key_frame 10 \
  --dataset cholecseg_sub \
  --ply_name ascii_frame_0_thickness.ply \
  --stage_num 10

# cholecseg_sub/video01_00240
CUDA_VISIBLE_DEVICES=1 python simulation_gt.py \
  --model_path ./model/cholecseg_sub/video01_00240 \
  --n_epoch 50 \
  --output_path ./model/cholecseg_sub/video01_00240 \
  --physics_config ./config/cholecseg_sub/video01_00240_config_gt.json \
  --n_key_frame 10 \
  --dataset cholecseg_sub \
  --ply_name ascii_frame_0_thickness.ply \
  --stage_num 10


# cholecseg_sub/video01_00240
CUDA_VISIBLE_DEVICES=1 python simulation_gt.py \
  --model_path ./model/cholecseg_sub/video17_01803 \
  --n_epoch 50 \
  --output_path ./model/cholecseg_sub/video17_01803 \
  --physics_config ./config/cholecseg_sub/video17_01803_config_gt.json \
  --n_key_frame 10 \
  --dataset cholecseg_sub \
  --ply_name ascii_frame_0_thickness.ply \
  --stage_num 10

# endonerf/pulling
CUDA_VISIBLE_DEVICES=1 python simulation_gt.py \
  --model_path ./model/endonerf/pulling_soft_tissues \
  --n_epoch 50 \
  --output_path ./model/endonerf/pulling_soft_tissues \
  --physics_config ./config/endonerf/pulling_soft_tissues_config_gt_test_ys.json \
  --n_key_frame 10 \
  --dataset endonerf \
  --ply_name ascii_frame_0_thickness.ply \
  --stage_num 10


# endonerf/cutting
CUDA_VISIBLE_DEVICES=1 python simulation_gt.py \
  --model_path ./model/endonerf/cutting_tissues_twice \
  --n_epoch 50 \
  --output_path ./model/endonerf/cutting_tissues_twice \
  --physics_config ./config/endonerf/cutting_tissues_twice_config_gt.json \
  --n_key_frame 10 \
  --dataset endonerf \
  --ply_name ascii_frame_0_thickness.ply \
  --stage_num 10

# porcine/gallbladder
CUDA_VISIBLE_DEVICES=1 python simulation_gt.py \
  --model_path ./model/porcine_endo/gallbladder \
  --n_epoch 50 \
  --output_path ./model/porcine_endo/gallbladder \
  --physics_config ./config/porcine_endo/gallbladder_config_gt.json \
  --n_key_frame 10 \
  --dataset porcine_endo \
  --ply_name point_cloud.ply \
  --stage_num 10 \
  --save_debug_flow # optional

CUDA_VISIBLE_DEVICES=1 python simulation_gt.py \
  --model_path ./model/porcine_endo/gallbladder \
  --output_path ./model/porcine_endo/gallbladder \
  --physics_config ./config/porcine_endo/gallbladder_config_gt.json \
  --n_key_frame 10 \
  --dataset porcine_endo \
  --save_debug_flow # optional

# porcine/liver
CUDA_VISIBLE_DEVICES=1 python simulation_gt.py \
  --model_path ./model/porcine_endo/liver \
  --n_epoch 50 \
  --output_path ./model/porcine_endo/liver \
  --physics_config ./config/porcine_endo/liver_config_gt.json \
  --n_key_frame 10 \
  --dataset porcine_endo \
  --ply_name ascii_frame_0_thickness.ply \
  --stage_num 10

# porcine/stomach
CUDA_VISIBLE_DEVICES=1 python simulation_gt.py \
  --model_path ./model/porcine_endo/stomach \
  --n_epoch 50 \
  --output_path ./model/porcine_endo/stomach \
  --physics_config ./config/porcine_endo/stomach_config_gt.json \
  --n_key_frame 10 \
  --dataset porcine_endo \
  --ply_name ascii_frame_0_thickness.ply \
  --stage_num 10

# bird
CUDA_VISIBLE_DEVICES=1 python simulation_gt.py \
  --model_path ./model/pacnerf/bird_test \
  --n_epoch 50 \
  --output_path ./model/pacnerf/bird_test \
  --physics_config ./config/pacnerf/bird_config_gt.json \
  --n_key_frame 13 \
  --dataset pacnerf \
  --ply_name point_cloud.ply



CUDA_VISIBLE_DEVICES=1 python simulation_gt.py \
  --model_path ./model/endonerf/pulling_soft_tissues \
  --n_epoch 50 \
  --output_path ./model/endonerf/pulling_soft_tissues_change_view_point \
  --physics_config ./config/endonerf/pulling_soft_tissues_config_gt_test_ys.json \
  --n_key_frame 10 \
  --dataset endonerf \
  --ply_name ascii_frame_0_thickness.ply \
  --stage_num 10

CUDA_VISIBLE_DEVICES=1 python simulation_gt.py \
--model_path ./model/porcine_endo/ballbladder \
--n_epoch 50 \
--output_path ./model/porcine_endo/ballbladder_change_view_point \
--physics_config ./config/porcine_endo/ballbladder_config_gt_change_viewpoint.json \
--n_key_frame 10 \
--dataset porcine_endo \
--ply_name ascii_frame_0_thickness.ply \
--stage_num 10