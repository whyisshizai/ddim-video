import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from copy import deepcopy

from .ema import EMA
from .utils import extract


class GaussianDiffusion(nn.Module):
    __doc__ = r"""Gaussian Diffusion model. Forwarding through the module returns diffusion reversal scalar loss tensor.

    Input:
        x:  tensor of shape (N, img_channels, *img_size)
        y:  tensor of shape (N)
    Output:
        scalar loss tensor
    Args:
        model (nn.Module): model which estimates diffusion noise
        video_size (tuple): image size tuple (H, W)
        video_channels (int): number of image channels
        video_length(int):length of video frames
        betas (np.ndarray): numpy array of diffusion betas
        loss_type (string): loss type, "l1" or "l2"
        ema_decay (float): model weights exponential moving average decay
        ema_start (int): number of steps before EMA
        ema_update_rate (int): number of steps before each EMA update
    """

    def __init__(
            self,
            model,
            video_size,
            video_channels,
            video_length,
            num_classes,
            betas,
            loss_type="l2",
            ema_decay=0.9999,
            ema_start=5000,
            ema_update_rate=1,
    ):
        super().__init__()

        self.model = model
        self.ema_model = deepcopy(model)

        self.ema = EMA(ema_decay)
        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.ema_update_rate = ema_update_rate
        self.step = 0

        self.video_size = video_size
        self.video_channels = video_channels
        self.video_length = video_length


        self.num_classes = num_classes
        if loss_type not in ["l1", "l2"]:
            raise ValueError("__init__() got unknown loss type")

        self.loss_type = loss_type
        self.num_timesteps = len(betas)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))
        # 隐式去噪模型ddim更改
        self.register_buffer("reciprocal_sqrt_alphas_cumprod", to_torch(np.sqrt(1 / alphas_cumprod)))

        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))

        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                self.ema.update_model_average(self.ema_model, self.model)

    @torch.no_grad()
    def remove_noise(self, x, t, y, use_ema=True,prompt=None):
        alpha_t = extract(self.reciprocal_sqrt_alphas, t, x.shape)

        if use_ema:
            return (
                # 按照马尔可夫链进行去噪
                    (x - extract(self.remove_noise_coeff, t, x.shape) * self.ema_model(x, t, y)) * alpha_t
            )
        else:
            return (
                    (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t, y,prompt)) * alpha_t
            )

    @torch.no_grad()
    # ddim的隐式去噪模型
    def implicit_remove_noise(self, x, t, y, use_ema=True, eta=None,prompt=None):
        """
        :param eta: 超参数eta，控制不确定度的值
        :param t: 时间步
        :param y: 附加值
        :param use_ema: 使用ema_model
        :param x: 加噪后的图像，现在对其进行去噪
        :return: 返回去噪后的图像张量
        """
        # DDIM: x_{t-1} = sqrt(ᾱ_{t-1}) * (x_t/sqrt(ᾱ_t) - epsilon * (1-ᾱ_t)/(sqrt(1-ᾱ_t))) + sigma_t * epsilon
        # 其中在确定性情况下 sigma_t = 0

        # a_bar~
        sqrt_alpha_bar_t = extract(self.sqrt_alphas_cumprod, t, x.shape)  # sqrt(ᾱ_t)

        # 1- a_bar
        sqrt_one_minus_alpha_bar_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)  # sqrt(1 - ᾱ_t)
        one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t ** 2  # 1 - ᾱ_t
        # 得到上一时间步
        prev_t = torch.clamp(t - 1, min=0)  # 确保 t-1 不小于 0
        sqrt_alpha_bar_prev_t = extract(self.sqrt_alphas_cumprod, prev_t, x.shape)  # sqrt(ᾱ_{t-1})
        one_minus_alpha_bar_prev_t = extract(self.sqrt_one_minus_alphas_cumprod, prev_t, x.shape) ** 2  # 1 - ᾱ_{t-1}
        # στi(eta) = eta * sqrt[ (1 − ατi−1) / (1 − ατi) ] * sqrt[ 1 − ( ατi / ατi−1 ) ]

        # 处理超参数 eta，计算 sigma_t
        if eta is None:
            sigma_t = 0  # 确定性采样
        else:
            sigma_t = eta * torch.sqrt(one_minus_alpha_bar_prev_t / one_minus_alpha_bar_t) * sqrt_one_minus_alpha_bar_t

        # 预测噪声
        if use_ema:
            epsilon = self.ema_model(x, t, y)
        else:
            epsilon = self.model(x, t, y, prompt=prompt)
        # 隐式去噪公式
        return sqrt_alpha_bar_prev_t * (
                x / sqrt_alpha_bar_t - epsilon * one_minus_alpha_bar_t / sqrt_one_minus_alpha_bar_t) + sigma_t * epsilon

    @torch.no_grad()
    def sample(self, batch_size, device, x=None, y=None, use_ema=True,prompt= None):
        print(f'sample procssing output size :{self.video_size}')
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        if x is None:
            x = torch.randn(batch_size, self.video_channels, 1, *self.video_size, device=device)
            # 注意不要让y被覆盖掉了
            for i in range(3):
                x = torch.cat([x, x.clone().detach()+torch.randn_like(x)], dim=2)

        for t in range(self.num_timesteps - 1, -1, -1):
            if not t % 10: print(t)
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            # 去噪
            x = self.remove_noise(x, t_batch, y, use_ema, prompt=prompt)

            if t > 0:
                # 加噪
                
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

        return x.cpu().detach()

    @torch.no_grad()
    def implicit_sample(self, batch_size, device, x=None, y=None, prompt=None,use_ema=True, eta=None, time_steps=None):
        print(f'implicit_sample procssing output size :{self.video_size},length:{self.video_length}')
        # if y is not None:
        #     if batch_size != y:
        #         raise ValueError("sample batch size different from length of given y")

        if x is None:
            x = torch.randn(batch_size, self.video_channels,1, *self.video_size, device=device)
            #注意不要让y被覆盖掉了
            if y is None:
                for i in range(3):
                    y = x.clone().detach()
                    x = torch.cat([x, y+torch.randn_like(x)], dim=2)
            else:
                for i in range(3):
                    x = torch.cat([x, x.clone()+torch.randn_like(x)], dim=2)

        if time_steps is None: time_steps = self.num_timesteps

        for t in range(time_steps - 1, -1, -1):
            if not t % 100: print(t)
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            # 去噪
            # eta确定性参数
            x = self.implicit_remove_noise(x, t_batch, y, use_ema, prompt=prompt, eta=eta)

            if t > 0:
                # 加噪
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

        return x.cpu().detach()

    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, y=None, use_ema=True):
        """
        采样扩散序列
        Args:
            batch_size: 批次大小
            device: 设备
            y: 条件标签
            use_ema: 是否使用EMA模型
        Returns:
            diffusion_sequence: 扩散序列列表
        """
        if y is not None and batch_size != len(y):
            raise ValueError("采样批次大小与给定的y长度不匹配")

        # 初始化随机噪声
        x = torch.randn(batch_size, self.video_channels, 1, *self.video_size, device=device)
        if y is None:
            for i in range(3):
                y = x.clone().detach() + torch.randn_like(x)
                x = torch.cat([x, y], dim=2)
        else:
            for i in range(3):
                x = torch.cat([x, x.clone()+torch.randn_like(x)], dim=2)
        diffusion_sequence = [x.cpu().detach()]

        # 反向扩散过程
        for t in range(self.num_timesteps - 1, -1, -1):
            if not t % 100:
                print(f'当前时间步:{t}')
                
            # 创建时间步batch
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            
            # 去噪步骤
            x = self.remove_noise(x, t_batch, y, use_ema)

            # 如果不是最后一步,添加噪声
            if t > 0:
                noise = torch.randn_like(x)
                sigma = extract(self.sigma, t_batch, x.shape)
                x = x + sigma * noise

            # 保存当前状态
            diffusion_sequence.append(x.cpu().detach())

        return diffusion_sequence

    def implicit_sample_diffusion_sequence(self, batch_size, device, y=None, use_ema=True, time_steps=None):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.video_channels, 1, *self.video_size, device=device)
        if y is None:
            for i in range(3):
                y = x.clone().detach()
                x = torch.cat([x, y+torch.randn_like(x)], dim=2)
        else:
            for i in range(3):
                x = torch.cat([x, x.clone()+torch.randn_like(x)], dim=2)
        diffusion_sequence = [x.cpu().detach()]
        if time_steps is None: time_steps = self.num_timesteps
        for t in range(time_steps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.implicit_sample(x, t_batch, y, use_ema)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

            diffusion_sequence.append(x.cpu().detach())

        return diffusion_sequence


    def perturb_x(self, x, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )

    def get_losses(self, x, t, y,prompt = None):
        noise = torch.randn_like(x)
        perturbed_x = self.perturb_x(x, t, noise)
        estimated_noise = self.model(perturbed_x, t, y, prompt)
        if self.loss_type == "l1":
            loss = F.l1_loss(estimated_noise, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(estimated_noise, noise)

        return loss

    def forward(self, x, y=None,prompt = None):
        b,c,t,h,w = x.shape  # 传入视频的大小

        device = x.device

        if h != self.video_size[0]:
            print(f'传入的高 h{h} and the img size{self.video_size[0]} is wrong')
            raise ValueError("image height does not match diffusion parameters")
        if w != self.video_size[0]:
            raise ValueError("image width does not match diffusion parameters")

        t = torch.randint(0, self.num_timesteps, (b,), device=device)

        #x:[bs,c,t,h,w]
        #
        return self.get_losses(x, t, y,prompt)


def generate_cosine_schedule(T, s=0.008):
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2

    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0)

    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))

    return np.array(betas)


def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)