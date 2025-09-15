import math
from functools import partial
from inspect import isfunction

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) /
                n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, time_size, channels=3, loss_type='l1',
                 conditional=True, schedule_opt=None):

        super().__init__()
        self.channels = channels
        self.time_size = time_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        if schedule_opt is not None:
            pass

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])

        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
                             (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
               self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
                         x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_temp = torch.cat([condition_x, x], dim=1)
            noise = self.denoise_fn(x_temp, noise_level)
            x_recon = self.predict_start_from_noise(x, t=t, noise=noise)
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(self.min_num, self.max_num)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)

        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    # x_in self.data['SR']
    def p_sample_loop(self, x_in, observed_mask,anomaly_scores,continous=False):
        device = self.betas.device
        q = 0
        batch_size, channel, seq_len, feature_dim = x_in.shape
        # 超参数设置
        alpha = 1.0  # 公式4中的α参数
        lambda_val = 0.1  # 公式9中的λ参数
        N0 = 1.0  # 公式9中的N0参数
        kappa = 0.5  # 公式5中的κ参数

        # 计算观测点的自适应权重（公式4）
        # anomaly_scores形状应为 (batch_size, seq_len) 或 (batch_size, channel, seq_len)
        # 我们需要确保权重与x_in的形状匹配
        if anomaly_scores.dim() == 2:  # (batch_size, seq_len)
            # 扩展维度以匹配x_in
            weights = torch.exp(-alpha * anomaly_scores).unsqueeze(1).unsqueeze(-1)
            weights = weights.expand(-1, channel, -1, feature_dim)
        else:  # 假设形状为 (batch_size, channel, seq_len)
            weights = torch.exp(-alpha * anomaly_scores).unsqueeze(-1)
            weights = weights.expand(-1, -1, -1, feature_dim)

        # 创建观测点掩码（observed_mask中0表示观测点）
        obs_mask = (observed_mask == 0).float()

        # 应用权重到观测点
        weighted_obs = x_in * obs_mask.unsqueeze(-1).expand_as(x_in) * weights

        # 准备初始噪声（公式5）
        shape = x_in.shape
        noise = torch.randn(shape, device=device)

        # 结合观测点和噪声（公式5）
        x_t = kappa * weighted_obs + (1 - kappa) * noise
        # 如果需要连续输出，初始化返回张量
        if continous:
            ret_img = [x_t]
        sample_inter = (1 | (self.num_timesteps // 10))

        # 反向扩散过程
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            # 使用加权的观测点作为条件进行采样
            x_t_minus_1 = self.p_sample(
                x_t, i,
                condition_x=weighted_obs,  # 使用加权的观测点作为条件
                clip_denoised=True
            )

            # 动态权重平滑（公式9和10）
            h = N0 * torch.exp(torch.tensor(-lambda_val * i))  # 公式9

            # 应用动态权重平滑（公式10）
            # 注意：这里s是观测点掩码，1表示观测点，0表示非观测点
            s = obs_mask.unsqueeze(-1).expand_as(x_t_minus_1)
            x_t_minus_1 = (
                    s * ((1 - weights * h) * x_t_minus_1 + weights * h * x_in) +
                    (1 - s) * x_t_minus_1
            )

            x_t = x_t_minus_1

            # 如果需要连续输出，保存当前步骤
            if continous and i % sample_inter == 0:
                ret_img.append(x_t)

        if continous:
            return torch.stack(ret_img)
        else:
            return x_t

    @torch.no_grad()
    def sample(self, observed_mask,anomaly_scores,batch_size=1, continous=False):
        time_size = self.time_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, time_size, time_size), continous)

    @torch.no_grad()
    # self.netG.module.super_resolution(
    #                     self.data['SR'], continous=continous, min_num=min_num, max_num=max_num)
    def super_resolution(self, x_in,observed_mask,anomaly_scores,min_num, max_num, continous=False):
        self.min_num = min_num
        self.max_num = max_num
        return self.p_sample_loop(x_in,observed_mask,anomaly_scores, continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
                continuous_sqrt_alpha_cumprod * x_start +
                (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)

        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t - 1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)

        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_cat = torch.cat([x_in['SR'], x_noisy], dim=1)
            x_recon = self.denoise_fn(x_cat, continuous_sqrt_alpha_cumprod)

        loss = self.loss_func(noise, x_recon)
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
