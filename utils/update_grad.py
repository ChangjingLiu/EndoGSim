import torch
import numpy as np
import warp as wp
from mpm_solver_warp.mpm_utils import update_param
import math

def update_grad_param(param, param_grad, n_particles, lrate=1.0, lower=-1.0, upper=-0.4, gn=False, scale=1., log_name=None, debug=False):
    grad = wp.to_torch(param_grad) / scale
    
    if gn:
        max_grad, min_grad = torch.max(grad), torch.min(grad)
        grad = (grad - min_grad) / (max_grad - min_grad) - 0.5 if max_grad - min_grad != 0 else torch.zeros_like(grad)

    if not debug:
        wp.launch(update_param, n_particles, [param, wp.from_torch(grad), lrate, upper, lower])

    if log_name is not None:
        print(f"- {log_name}: {torch.mean(wp.to_torch(param)).item()}, grad_{log_name}: {torch.mean(grad).item()}, lr_{log_name}: {lrate}")


def lr_scheduler(lr_init, lr_end, step, total_steps, warmup_steps=0, max_steps=None):
    if max_steps is None:
        max_steps = total_steps
    if step < warmup_steps:
        lr = float(step) / float(warmup_steps) * (lr_init - lr_end) + lr_end
    elif step < max_steps:
        lr = lr_end + 0.5 * (lr_init - lr_end) * (1 + np.cos((step - warmup_steps) / (max_steps - warmup_steps) * np.pi))
    else:
        lr = lr_end
    return lr

def update_grad_param_limit(
    param,
    param_grad,
    n_particles,
    lrate=1.0,
    lower=-1.0,
    upper=-0.4,
    gn=False,
    scale=1.0,
    log_name=None,
    debug=False,
    clip_min=None,  # 新增裁剪下界
    clip_max=None,   # 新增裁剪上界
    SCALE=1e6  # 新增杨氏模量缩放因子
):
    # 1. warp tensor -> torch tensor
    grad = wp.to_torch(param_grad) / scale

    # 杨氏模量信息
    param_torch = wp.to_torch(param)
    param_mean = param_torch.mean().item()

    logE_old = math.log10(param_mean)
    grad_logE = grad.mean().item() * param_mean * math.log(10)
    
    # ===== E 的 log-space 信息 =====
    if log_name is not None and "E" in log_name:
        print(f"[Original grad_{log_name}] mean: {grad.mean().item()}, max: {grad.max().item()}, min: {grad.min().item()}")

        # print(f"[logE info]")
        # print(f"  log10(E_old) = {logE_old}")
        # print(f"  grad_logE    = {grad_logE}")
        # print("log10(E_new) ≈ log10(E_old) - lr * grad_logE")
        # #============================

    # 2. 可选归一化（GN）
    if gn:
        max_grad, min_grad = torch.max(grad), torch.min(grad)
        if max_grad - min_grad != 0:
            grad = (grad - min_grad) / (max_grad - min_grad) - 0.5
        else:
            grad = torch.zeros_like(grad)

    # 3. 梯度裁剪
    if clip_min is not None and clip_max is not None:
        grad = torch.clamp(grad, clip_min, clip_max)

    # 4. warp tensor -> 回写
    grad_warp = wp.from_torch(grad)

    # 5. 更新参数
    if not debug:
        wp.launch(update_param, n_particles, [param, grad_warp, lrate, upper, lower])

    # 6. 打印日志
    if log_name is not None:
        print(f"- {log_name}: {torch.mean(wp.to_torch(param)).item()}, "
              f"grad_{log_name}: {torch.mean(grad).item()}, lr_{log_name}: {lrate}")
    
    # 打印杨氏模量信息
    if log_name is not None and "E" in log_name:
        real_E = torch.mean(wp.to_torch(param)) * SCALE
        # print(f"[logE info]")
        # print(f"log10(E_old) = {logE_old}")
        # print(f"grad_logE    = {grad_logE}")
        # print(f"grad_logE after clipping: mean {grad.mean().item() * param_mean * math.log(10)}")
        # print(f"log10(E_new) ≈ {math.log10(real_E.item())}")

        # print("log10(E_new) ≈ log10(E_old) - lr * grad_logE")

        print(f"Trian {log_name}: {real_E.item()/1e3} kPa")
    
    if log_name is not None and "yield_stress" in log_name:
        real_ys = torch.mean(wp.to_torch(param)) * SCALE
        print(f"Trian {log_name}: {real_ys.item()/1e3} kPa")

    if log_name is not None and "plastic_viscosity" in log_name:
        real_pv = torch.mean(wp.to_torch(param)) * SCALE
        print(f"Trian {log_name}: {real_pv.item()/1e3} kPa·s")
