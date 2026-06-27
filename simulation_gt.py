
# 用于生成光流关键帧视频

import sys
import argparse
import math
import cv2
import torch
from torch import nn
import os
import numpy as np
from tqdm import tqdm
import point_cloud_utils as pcu

# Gaussian splatting dependencies
sys.path.append("gs")
from scene.gaussian_model import GaussianModel

# MPM dependencies
from mpm_solver_warp.engine_utils import *
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
import warp as wp

# Particle filling dependencies
from particle_filling.filling import *

# Utils
sys.path.append("utils")
from utils.decode_param import *
from utils.transformation_utils import *
from utils.camera_view_utils import *
from utils.render_utils import *
from utils.save_video import save_video

from video_distillation.cogv_guidance import CogVideoGuidance

torch.manual_seed(0)

wp.init()
wp.config.verify_cuda = True

# Reduce Taichi reserved GPU memory to leave room for PyTorch (was 8.0 GB)
# Lowering this helps prevent non-PyTorch allocations from filling the GPU.
ti.init(arch=ti.cuda, device_memory_fraction=0.1)


class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_checkpoint(model_path, iteration=-1, material=None):
    # Find checkpoint
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    if args.dataset == "endonerf" or args.dataset == "cholecseg_sub" or args.dataset == "porcine_endo":
        checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", args.ply_name
    )
    else:
        checkpt_path = os.path.join(
            checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
        )
    
    # sh_degree=0, if you use a 3D asset without spherical harmonics
    from plyfile import PlyData
    plydata = PlyData.read(checkpt_path)
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    
    # Load guassians
    sh_degree = int(math.sqrt((len(extra_f_names)+3) // 3)) - 1
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path, material)
    return gaussians


def load_inpaint_gs(model_path):
    checkpt_path = os.path.join(
        model_path, "inpaint_points.ply"
    )
    if not os.path.exists(checkpt_path):
        return None
    
    # sh_degree=0, if you use a 3D asset without spherical harmonics
    from plyfile import PlyData
    plydata = PlyData.read(checkpt_path)
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    
    # Load guassians
    sh_degree = int(math.sqrt((len(extra_f_names)+3) // 3)) - 1
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians


def render_frame(mpm_solver, gs_num, init_len, moving_pts_path, 
                current_camera, gaussians, params_inpaint,
                opacity, shs,
                unselected_pos, unselected_cov, unselected_opacity, unselected_shs):

    pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
    cov3D = mpm_solver.export_particle_cov_to_torch()
    rot = mpm_solver.export_particle_R_to_torch()
    
    cov3D = cov3D.view(-1, 6)[:gs_num].to(device)
    rot = rot.view(-1, 3, 3)[:gs_num].to(device)

    pos = pos[:init_len,:]
    pos = apply_inverse_rotations(
        undotransform2origin(
            undoshift2center111(pos), scale_origin, original_mean_pos
        ),
        rotation_matrices,
    )
    cov3D = cov3D / (scale_origin * scale_origin)
    cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
    if os.path.exists(moving_pts_path):
        pos = torch.cat([pos, unselected_pos], dim=0)
        cov3D = torch.cat([cov3D, unselected_cov], dim=0)
        opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
        shs = torch.cat([shs_render, unselected_shs], dim=0)
    if params_inpaint is not None:
        pos = torch.cat([pos, params_inpaint['pos']], dim=0)
        cov3D = torch.cat([cov3D, params_inpaint['cov3D_precomp']], dim=0)
        opacity = torch.cat([opacity, params_inpaint['opacity']], dim=0)
        shs = torch.cat([shs, params_inpaint['shs']], dim=0)
    if preprocessing_params["sim_area"] is not None:
        pos = torch.cat([pos, unselected_pos], dim=0)
        cov3D = torch.cat([cov3D, unselected_cov], dim=0)
        opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
        shs = torch.cat([shs_render, unselected_shs], dim=0)

    colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
    rendering, _, _, _ = rasterize(
        means3D=pos,
        means2D=init_screen_points,
        means2D_abs=pos,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=None,
        rotations=None,
        cov3D_precomp=cov3D,
    )
    return rendering


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--physics_config", type=str, required=True)
    parser.add_argument("--white_bg", type=bool, default=False)
    parser.add_argument("--output_ply", action="store_true")
    parser.add_argument("--output_h5", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_debug_flow", action="store_true", help="Save optical-flow debug images to debug_flow/")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--downsample", type=float, default=1.0)
    parser.add_argument("--n_epoch", type=int, default=1)
    parser.add_argument("--n_key_frame", type=int, default=1)
    parser.add_argument("--stage_num", type=int, default=10)
    # parser.add_argument("--endonerf", type=bool, default=False)
    parser.add_argument("--ply_name", type=str, default="point_cloud.ply")
    # parser.add_argument("--n_epoch", type=int, default=10)
    # parser.add_argument("--n_key_frame", type=int, default=8)
    parser.add_argument(
        "--dataset",
        type=str,
        default="endonerf",
        choices=["endonerf", "cholecseg_sub","pacnerf","porcine_endo"],
        help="Choose which dataset to use"
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        AssertionError("Model path does not exist!")
    if not os.path.exists(args.physics_config):
        AssertionError("Scene config does not exist!")
    

    if args.output_path is not None and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.debug:
        if not os.path.exists(f"{args.output_path}/log"):
            os.makedirs(f"{args.output_path}/log")

    # load scene config
    print("Loading scene config...")
    (
        material_params,
        bc_params,
        time_params,
        preprocessing_params,
        camera_params,
        _,
    ) = decode_param_json(args.physics_config)

    # load gaussians
    print("Loading gaussians...")
    model_path = args.model_path
    gaussians = load_checkpoint(model_path, material=material_params["material"])
    gaussians_inpaint = load_inpaint_gs(model_path)
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )

    # init the scene
    print("Initializing scene and pre-processing...")
    params = load_params_from_gs(gaussians, pipeline)
    params_inpaint = None
    if gaussians_inpaint is not None:
        params_inpaint = load_params_from_gs(gaussians_inpaint, pipeline)

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_screen_points = params["screen_points"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]

    # throw away low opacity kernels
    mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
    init_pos = init_pos[mask, :]
    init_cov = init_cov[mask, :]
    init_opacity = init_opacity[mask, :]
    init_screen_points = init_screen_points[mask, :]
    init_shs = init_shs[mask, :]
    
    # optimize moving parts only
    unselected_pos, unselected_cov, unselected_opacity, unselected_shs = (
        None,
        None,
        None,
        None,
    )

    moving_pts_path = os.path.join(model_path, "moving_part_points.ply")
    if os.path.exists(moving_pts_path):
        moving_pts = pcu.load_mesh_v(moving_pts_path)
        moving_pts = torch.from_numpy(moving_pts).float().to("cuda")
        thres = 0.5 / material_params["n_grid"]
        if "playdoh" in model_path:
            thres = 1.0 / material_params["n_grid"]
        freeze_mask = find_far_points(
            init_pos, moving_pts, thres=thres
        ).bool()
        moving_pts.to("cpu")
        unselected_pos = init_pos[freeze_mask, :]
        unselected_cov = init_cov[freeze_mask, :]
        unselected_opacity = init_opacity[freeze_mask, :]
        unselected_shs = init_shs[freeze_mask, :]

        init_pos = init_pos[~freeze_mask, :]
        init_cov = init_cov[~freeze_mask, :]
        init_opacity = init_opacity[~freeze_mask, :]
        init_shs = init_shs[~freeze_mask, :]


    # rorate and translate object
    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"],
    )
    print('rotatted_pos: ',rotation_matrices)
    rotated_pos = apply_rotations(init_pos, rotation_matrices)

    # 添加sim_area的处理
    if preprocessing_params["sim_area"] is not None:
        boundary = preprocessing_params["sim_area"]
        print('boundary: ',boundary)

        assert len(boundary) == 6
        mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool).to(device="cuda")
        for i in range(3):
            mask = torch.logical_and(mask, rotated_pos[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, rotated_pos[:, i] < boundary[2 * i + 1])

        unselected_pos = init_pos[~mask, :]
        unselected_cov = init_cov[~mask, :]
        unselected_opacity = init_opacity[~mask, :]
        unselected_shs = init_shs[~mask, :]

        rotated_pos = rotated_pos[mask, :]
        init_cov = init_cov[mask, :]
        init_opacity = init_opacity[mask, :]
        init_shs = init_shs[mask, :]

    scaling = 1.0
    if 'cat' in model_path:
        scaling = 0.7
    if 'letter' in model_path:
        scaling = 2.0
    if 'cream' in model_path:
        scaling = 0.8
    if 'toothpaste' in model_path:
        scaling = 0.6
    if 'playdoh' in model_path:
        scaling = 0.75
    transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos, scaling=scaling)
    transformed_pos = shift2center111(transformed_pos)
    print("original_mean_pos", original_mean_pos)
    print("scale_origin", scale_origin)

    # modify covariance matrix accordingly
    init_cov = apply_cov_rotations(init_cov, rotation_matrices)
    init_cov = scale_origin * scale_origin * init_cov

    if args.debug:
        particle_position_tensor_to_ply(
            transformed_pos,
            f"{args.output_path}/log/transformed_particles.ply",
        )

    # fill particles if needed
    gs_num = transformed_pos.shape[0]
    device = "cuda:0"
    filling_params = preprocessing_params["particle_filling"]

    if filling_params is not None:
        print("Filling internal particles...")
        mpm_init_pos = fill_particles(
            pos=transformed_pos,
            opacity=init_opacity,
            cov=init_cov,
            grid_n=filling_params["n_grid"],
            max_samples=filling_params["max_particles_num"],
            grid_dx=material_params["grid_lim"] / filling_params["n_grid"],
            density_thres=filling_params["density_threshold"],
            search_thres=filling_params["search_threshold"],
            max_particles_per_cell=filling_params["max_partciels_per_cell"],
            search_exclude_dir=filling_params["search_exclude_direction"],
            ray_cast_dir=filling_params["ray_cast_direction"],
            boundary=filling_params["boundary"],
            smooth=filling_params["smooth"],
        ).to(device=device)

        if args.debug:
            particle_position_tensor_to_ply(mpm_init_pos, f"{args.output_path}/log/filled_particles.ply")
    else:
        mpm_init_pos = transformed_pos.to(device=device)

    # init the mpm solver
    print("Initializing MPM solver and setting up boundary conditions...")
    mpm_init_vol = get_particle_volume(
        mpm_init_pos,
        material_params["n_grid"],
        material_params["grid_lim"] / material_params["n_grid"],
        unifrom=material_params["material"] == "sand",
    ).to(device=device)

    if filling_params is not None and filling_params["visualize"] == True:
        shs, opacity, mpm_init_cov = init_filled_particles(
            mpm_init_pos[:gs_num],
            init_shs,
            init_cov,
            init_opacity,
            mpm_init_pos[gs_num:],
        )
        _pos = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(mpm_init_pos[gs_num:]), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            )
        print("gs.xyz", gaussians._xyz.shape)
        gaussians._xyz = nn.Parameter(torch.tensor(torch.cat([gaussians._xyz, _pos], 0), dtype=torch.float, device="cuda").requires_grad_(True))
        _opacity = torch.zeros((_pos.shape[0], 1)).to("cuda:0")
        gaussians._opacity = nn.Parameter(torch.tensor(torch.cat([gaussians._opacity, _opacity], 0), dtype=torch.float, device="cuda").requires_grad_(True))
        _scaling = torch.zeros((_pos.shape[0], 3)).to("cuda:0")
        gaussians._scaling = nn.Parameter(torch.tensor(torch.cat([gaussians._scaling, _scaling], 0), dtype=torch.float, device="cuda").requires_grad_(True))
        _rotation = torch.zeros((_pos.shape[0], 4)).to("cuda:0")
        gaussians._rotation = nn.Parameter(torch.tensor(torch.cat([gaussians._rotation, _rotation], 0), dtype=torch.float, device="cuda").requires_grad_(True))

        gs_num = mpm_init_pos.shape[0]
    else:
        mpm_init_cov = torch.zeros((mpm_init_pos.shape[0], 6), device=device)
        mpm_init_cov[:gs_num] = init_cov
        shs = init_shs
        opacity = init_opacity


    # set up the mpm solver
    mpm_solver = MPM_Simulator_WARP(10)
    mpm_solver.load_initial_data_from_torch(
        mpm_init_pos,
        mpm_init_vol,
        mpm_init_cov,
        n_grid=material_params["n_grid"],
        grid_lim=material_params["grid_lim"],
    )
    mpm_solver.set_parameters_dict(material_params)

    if args.dataset == "endonerf" or args.dataset == "cholecseg_sub" or args.dataset == "porcine_endo":
        # 处理边界条件的坐标系转换，config需填写的是原始坐标系
        for bc in bc_params:
            if bc['type'] in ['particle_impulse', 'cuboid']:
                bc['point']= bc['point'] - original_mean_pos.detach().cpu().numpy()
                bc['point']= bc['point'] * scale_origin.detach().cpu().numpy()
                bc['point'] = bc['point'] + np.array([1.0, 1.0, 1.0])
            
            if bc['type'] in ['particle_velocity']:
                bc['size'] = bc['size'] * scale_origin.detach().cpu().numpy() # 施加力的size是重建尺寸
    set_boundary_conditions(mpm_solver, bc_params, time_params)
    mpm_solver.finalize_mu_lam()

    # camera setting
    mpm_space_viewpoint_center = (
        torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    )
    mpm_space_vertical_upward_axis = (
        torch.tensor(camera_params["mpm_space_vertical_upward_axis"])
        .reshape((1, 3))
        .cuda()
    )
    (
        viewpoint_center_worldspace,
        observant_coordinates,
    ) = get_center_view_worldspace_and_observant_coordinate(
        mpm_space_viewpoint_center,
        mpm_space_vertical_upward_axis,
        rotation_matrices,
        scale_origin,
        original_mean_pos,
    )

    # run the simulation
    
    # endonerf/pulling_soft_tissues
    if args.dataset == "endonerf":
        current_camera = get_camera_view_endonerf()
    elif args.dataset == "cholecseg_sub":
        current_camera = get_camera_view_cholecseg_sub()
    elif args.dataset == "porcine_endo":
        current_camera = get_camera_view_porcine_endo()
    else: 
        current_camera, _ = get_camera_view(
        model_path,
        center_view_world_space=viewpoint_center_worldspace,
        observant_coordinates=observant_coordinates,
        default_camera_index=camera_params["default_camera_index"],
        downsample=args.downsample
    )
    rasterize = initialize_resterize(
        current_camera, gaussians, pipeline, background
    )

    ## To render the first frame as image prompt
    opacity_render = opacity
    shs_render = shs
    init_len = mpm_init_pos.shape[0]
    image_prompt = render_frame(mpm_solver, gs_num, init_len, moving_pts_path, 
                                current_camera, gaussians, params_inpaint,
                                opacity_render, shs_render,
                                unselected_pos, unselected_cov, unselected_opacity, unselected_shs)

    
    # optimization settings
    substep_dt = time_params["substep_dt"] # 表示每个子步（substep）的时间步长
    frame_dt = time_params["frame_dt"] # 表示每帧的时间步长
    # opt_frame_dt = time_params["opt_frame_dt"] # 表示每个优化帧的时间步长
    step_per_frame = int(frame_dt / substep_dt) # 表示每帧包含的子步数量
    # step_per_opt_frame = int(opt_frame_dt / substep_dt) # 表示每个优化帧包含的子步数量
    # step_per_opt_frame = 1

    stage_num = args.stage_num #把整个优化过程切成多少个阶段，每个阶段负责不同的物理帧段（不同的 keyframes）
    frame_per_stage = args.n_key_frame # 每个阶段中处理的帧数

    ############# simulation

    image_generated = []
    image_generated.append(image_prompt)
    image_flow = []
    cnt =0
    for frame in tqdm(range(stage_num * frame_per_stage)):
        delta_r = camera_params["delta_r"]

        # endonerf/pulling_soft_tissues
        if args.dataset == "endonerf":
            current_camera = get_camera_view_endonerf()
        elif args.dataset == "cholecseg_sub":
            current_camera = get_camera_view_cholecseg_sub()
        elif args.dataset == "porcine_endo":
            current_camera = get_camera_view_porcine_endo()
        else:
            current_camera, _ = get_camera_view(
            model_path,
            default_camera_index=camera_params["default_camera_index"],
            center_view_world_space=viewpoint_center_worldspace,
            observant_coordinates=observant_coordinates,
            current_frame=frame,
            move_camera=camera_params["move_camera"],
            delta_a=camera_params["delta_a"] ,
            delta_e=camera_params["delta_e"] ,
            delta_r=delta_r,
            downsample=args.downsample
        )
        rasterize = initialize_resterize(
            current_camera, gaussians, pipeline, background
        )
        for _ in range(step_per_frame):
            mpm_solver.p2g2p(frame, substep_dt, device=device)
            cnt=cnt+1

        pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
        cov3D = mpm_solver.export_particle_cov_to_torch()
        rot = mpm_solver.export_particle_R_to_torch()

        cov3D = cov3D.view(-1, 6)[:gs_num].to(device)
        rot = rot.view(-1, 3, 3)[:gs_num].to(device)

        pos = pos[:init_len,:]
        pos = apply_inverse_rotations(
            undotransform2origin(
                undoshift2center111(pos), scale_origin, original_mean_pos
            ),
            rotation_matrices,
        )
        cov3D = cov3D / (scale_origin * scale_origin)
        cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
        opacity = opacity_render
        shs = shs_render
        if os.path.exists(moving_pts_path):
            pos = torch.cat([pos, unselected_pos], dim=0)
            cov3D = torch.cat([cov3D, unselected_cov], dim=0)
            opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
            shs = torch.cat([shs_render, unselected_shs], dim=0)
        if params_inpaint is not None:
            pos = torch.cat([pos, params_inpaint['pos']], dim=0)
            cov3D = torch.cat([cov3D, params_inpaint['cov3D_precomp']], dim=0)
            opacity = torch.cat([opacity, params_inpaint['opacity']], dim=0)
            shs = torch.cat([shs, params_inpaint['shs']], dim=0)
        if preprocessing_params["sim_area"] is not None:
            pos = torch.cat([pos, unselected_pos], dim=0)
            cov3D = torch.cat([cov3D, unselected_cov], dim=0)
            opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
            shs = torch.cat([shs_render, unselected_shs], dim=0)
        colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
        rendering, _, _, _ = rasterize(
            means3D=pos,
            means2D=init_screen_points,
            means2D_abs=init_screen_points,
            shs=None,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=None,
            rotations=None,
            cov3D_precomp=cov3D,
        )

        if cnt % (step_per_frame * stage_num) == 0:
            image_generated.append(rendering)
            image_flow.append(rendering)
            print("关键帧substep步数",cnt)

        cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        assert args.output_path is not None
        os.makedirs(os.path.join(args.output_path, 'frames'), exist_ok=True)
        cv2.imwrite(
            os.path.join(args.output_path, f"frames/{frame:04d}.png"),
            255 * cv2_img,
        )

    for i, img in enumerate(image_generated):
        os.makedirs(os.path.join(args.output_path, 'images_generated'), exist_ok=True)
        cv2_img = img.permute(1, 2, 0).detach().cpu().numpy()
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(
            os.path.join(args.output_path, f"images_generated/keyframe_{i*stage_num:04d}.png"),
            255 * cv2_img,
        )

    guidance_path = os.path.join(args.model_path, 'images_generated')
    if args.save_debug_flow:
        if not os.path.exists(guidance_path):
            raise AssertionError("Guidance frames do not exist!")

        # 光流初始化时候，计算参考光流
        guidance = CogVideoGuidance(guidance_path, downsample=args.downsample, num_frames=args.n_key_frame)

        # save log of optical flows
        img_list = torch.stack(image_flow)
        # 原始图像尺寸（以第一帧为准）
        orig_h, orig_w = image_flow[0].shape[-2], image_flow[0].shape[-1]
        _, flow_imgs = guidance.predict_flow(img_list, image_prompt.unsqueeze(0))
        os.makedirs(os.path.join(args.output_path, 'debug_flow'), exist_ok=True)
        for i, flow_img in enumerate(flow_imgs):
            cv2_img = flow_img.permute(1, 2, 0).detach().cpu().numpy()
            # resize 回原始尺寸
            cv2_img = cv2.resize(
                cv2_img,
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR
            )
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args.output_path, f"debug_flow/keyframe_{i*stage_num:04d}.png"), cv2_img)

    save_video(os.path.join(args.output_path, 'frames'), os.path.join(args.output_path, 'video_final.mp4'))