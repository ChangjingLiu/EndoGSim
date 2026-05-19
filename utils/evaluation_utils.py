import torch

def compute_ecms(velocities, alpha=0.1):
    """
    velocities: (T, N, 3)
    返回一个平均化后的 ECMS，不随粒子数/时间长度爆炸
    """

    velocities = velocities.float()
    T, N, C = velocities.shape

    if T < 3:
        raise ValueError("ECMS 需要至少 3 个时间步")

    # ---- 1. 一阶差分（平均化）----
    vel_diff = velocities[1:] - velocities[:-1]              # (T-1, N, 3)
    velocity_change_term = torch.mean(vel_diff ** 2)         # <---- mean！

    # ---- 2. 二阶差分（平均化）----
    vel_second = velocities[2:] - 2*velocities[1:-1] + velocities[:-2]
    velocity_grad_term = torch.mean(vel_second ** 2)         # <---- mean！

    # ---- 3. 速度范数（平均化）----
    velocity_norm = torch.mean(torch.norm(velocities, dim=-1))
    normalization_term = alpha * velocity_norm               # 或 alpha * mean(norm)

    ecms = velocity_change_term + velocity_grad_term + normalization_term

    print(f"ECMS(mean): {ecms.item()} -> "
          f"({velocity_change_term.item()}, {velocity_grad_term.item()}, {normalization_term.item()})")

    return ecms
