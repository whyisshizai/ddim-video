import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import GroupNorm
from torch.nn.functional import scaled_dot_product_attention as sdp


def get_norm(norm, num_channels, num_groups):
    if norm == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm == "gn":
        return nn.GroupNorm(num_groups, num_channels)
    elif norm is None:
        return nn.Identity()
    else:
        raise ValueError("unknown normalization type")


class PositionalEmbedding(nn.Module):
    """位置编码模块,用于计算时间步的位置嵌入。

    这个模块实现了Transformer中的位置编码机制,用于为时间步添加位置信息。
    它使用正弦和余弦函数的组合来生成独特的位置编码。

    工作原理:
    1. 输入时间步 x 会被缩放(乘以scale)
    2. 将维度dim分成两半,一半用于sin编码,一半用于cos编码
    3. 对每个维度使用不同频率的正弦波:
       - 频率随维度指数衰减: exp(-log(10000) * i / (dim/2))
       - i 是维度索引从0到dim/2-1
    4. 对缩放后的时间步和频率做外积运算,得到编码矩阵
    5. 将矩阵的正弦部分和余弦部分在最后一维拼接
    
    这种编码方式的优点是:
    - 每个时间步都有唯一的编码
    - 编码值的范围有界([-1,1])
    - 不同维度捕获不同尺度的时间信息
    - 编码可以平滑插值

    输入:
        x: 形状为 (N) 的张量,表示时间步
    输出:
        形状为 (N, dim) 的张量,表示位置编码
    参数:
        dim (int): 嵌入维度,必须为偶数
        scale (float): 应用于时间步的线性缩放因子。默认值: 1.0
    """

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0  # 确保维度为偶数 assert是断言错误，如果不为1那么报错
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim  # 计算基础频率
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  # 生成不同频率
        emb = torch.outer(x * self.scale, emb)  # 计算时间步与频率的组合
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # 拼接sin和cos结果
        return emb


class Downsample(nn.Module):
    __doc__ = r"""Downsamples a given tensor by a factor of 2. Uses strided convolution. Assumes even height and width.

    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: ignored
        y: ignored
    Output:
        tensor of shape (N, in_channels, H // 2, W // 2)
    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, in_channels):
        super().__init__()

        self.downsample = nn.Conv3d(in_channels, in_channels, 3, stride=2, padding=1)

    def forward(self, x, time_emb, y):

        #x[bs,c,t,h,w]
        if x.shape[2] % 2 != 0:
            raise ValueError(f"此处的t{x.shape[2]}不是偶数")
        if x.shape[3] % 2 == 1:
            raise ValueError("此处的h不是偶数")
        if x.shape[4] % 2 == 1:
            raise ValueError("此处的w不是偶数")

        return self.downsample(x)


class Upsample(nn.Module):
    __doc__ = r"""Upsamples a given tensor by a factor of 2. Uses resize convolution to avoid checkerboard artifacts.

    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: ignored
        y: ignored
    Output:
        tensor of shape (N, in_channels, H * 2, W * 2)
    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, in_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv3d(in_channels, in_channels, 3, padding=1),
        )

    def forward(self, x, time_emb, y):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    __doc__ = r"""
    IN:
        x: (B, in_channels, T, H, W)
        norm : group_norm
        num_groups (int): 默认: 32
    OUT:
        (B, in_channels, T,  H, W+)
    """
    def __init__(self, in_channels, norm="gn", num_groups=32,device = None):
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        # 获取归一化层
        self.norm = get_norm(norm, in_channels, num_groups)
        # 空间QKV和投影
        self.to_qkv_spatial = nn.Conv3d(in_channels, in_channels * 3, 1)  #空间Q,K,V
        # 时间QKV和投影
        self.to_qkv_temporal = nn.Conv1d(in_channels, in_channels * 3, 1)  # 时间维度的QKV
    def forward(self, x, prompt_emb=None):
        b, c, t, h, w = x.shape
        x_spatial = self.norm(x)
        # 提取feature map
        avg_pool = nn.AdaptiveAvgPool3d((t, 1, 1)).to(self.device)
        f = avg_pool(x_spatial).reshape(b, c,-1).transpose(-1,-2)
        #b t c
        #感知特征提取器
        fc = nn.Sequential(
            nn.Linear(c, c//2), #压缩
            nn.ReLU(inplace=True),
            nn.Linear(c//2, c), #恢复
            nn.Sigmoid()  # 得到0-1的权重
        ).to("cuda")
        fm = fc(f).transpose(-1,-2).view(b,c,t,1,1)

        #空间过三维卷积
        qkv_s = self.to_qkv_spatial(x_spatial)  # 一次性生成QKV
        qkv_s = qkv_s.reshape(b, 3, self.in_channels, t, h, w)
        qs, ks, vs = qkv_s[:, 0], qkv_s[:, 1], qkv_s[:, 2]
        qs = qs.reshape(b, c, t, -1)  # [b, c, t,h*w]
        ks = ks.reshape(b, c, t, -1)
        vs = vs.reshape(b, c, t, -1)
        #b c h*w t
        # 时间过一维卷积得到单一维度
        qkv_t = self.to_qkv_temporal(x_spatial.reshape(b, c, -1))
        qkv_t = qkv_t.reshape(b, 3, self.in_channels, -1)
        #bs c t*h*w
        qt, kt, vt = qkv_t[:, 0], qkv_t[:, 1], qkv_t[:, 2]
        scale = 1.0 / math.sqrt(t)
        if prompt_emb is not None:
            #cross
            p_l= nn.Linear(prompt_emb.shape[-1],t * h * w).to(self.device)
            prompt = p_l(prompt_emb)  # [b, t*h*w]
            scale = 1.0 / math.sqrt(prompt.shape[-1])
            #q * k * v
            kt = sdp(kt, prompt, prompt, attn_mask=None, dropout_p=0.1)
        #cross
        #时间维度的k去和空间的q查询
        s1 = sdp(qs,kt.reshape(b, c, t, -1),vs).transpose(-2, -1).reshape(b, c, t, h, w)
        #时间的q去和空间上的k查询
        s2 = sdp(qt,ks.reshape(b, c, -1),vt).reshape(b, c, t, h, w)
        #residual block
        #回归到时间-时间，空间-空间，整体残差
        o = (s1 + s2) * fm +  x
        return o


class ResidualBlock(nn.Module):
    __doc__ = r"""Applies two conv blocks with resudual connection. Adds time and class conditioning by adding bias after first convolution.

    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: time embedding tensor of shape (N, time_emb_dim) or None if the block doesn't use time conditioning
        y: classes tensor of shape (N) or None if the block doesn't use class conditioning
    Output:
        tensor of shape (N, out_channels, H, W)
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        time_emb_dim (int or None): time embedding dimension or None if the block doesn't use time conditioning. Default: None
        num_classes (int or None): number of classes or None if the block doesn't use class conditioning. Default: None
        activation (function): activation function. Default: torch.nn.functional.relu
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
        use_attention (bool): if True applies AttentionBlock to the output. Default: False
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            dropout,
            time_emb_dim=None,
            num_classes=None,
            activation=F.relu,
            norm="gn",
            num_groups=32,
            use_attention=False,
            device = None
    ):
        super().__init__()

        self.activation = activation

        self.norm_1 = get_norm(norm, in_channels, num_groups)
        self.conv_1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)

        self.norm_2 = get_norm(norm, out_channels, num_groups)
        self.conv_2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
        )

        self.time_bias = nn.Linear(time_emb_dim, out_channels) if time_emb_dim is not None else None
        self.class_bias = nn.Embedding(num_classes, out_channels) if num_classes is not None else None

        self.residual_connection = nn.Conv3d(in_channels, out_channels,
                                             1) if in_channels != out_channels else nn.Identity()
        self.attention = nn.Identity() if not use_attention else AttentionBlock(out_channels, norm, num_groups,device=device)

    def forward(self, x, time_emb=None, y=None, prompt=None):
        out = self.activation(self.norm_1(x))
        out = self.conv_1(out)

        if self.time_bias is not None:
            if time_emb is None:
                raise ValueError("time conditioning was specified but time_emb is not passed")
            #x[1,128,27,128,128]
            #channle down
            #这里报错是因为根本上这个time_bias是为了2D image设计的
            o = self.time_bias(self.activation(time_emb))
            #第一个维度bs不变，第二个维度channle不变,补充t,h,w None
            out += o[:, :, None, None,None] #在这里添加一个NOne

        if self.class_bias is not None:
            if y is None:
                raise ValueError("class conditioning was specified but y is not passed")

            out += self.class_bias(y)[:, :, None, None,None]

        out = self.activation(self.norm_2(out))
        out = self.conv_2(out) + self.residual_connection(x)
        #out[1,128,75,128,128]
        if isinstance(self.attention, nn.Identity):
            out = self.attention(out)
        else:
            out = self.attention(out, prompt)
        return out


class UNet(nn.Module):
    __doc__ = """UNet model used to estimate noise.

    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: time embedding tensor of shape (N, time_emb_dim) or None if the block doesn't use time conditioning
        y: classes tensor of shape (N) or None if the block doesn't use class conditioning
    Output:
        tensor of shape (N, out_channels, H, W)
    Args:
        img_channels (int): number of image channels
        base_channels (int): number of base channels (after first convolution)
        channel_mults (tuple): tuple of channel multiplers. Default: (1, 2, 4, 8)
        time_emb_dim (int or None): time embedding dimension or None if the block doesn't use time conditioning. Default: None
        time_emb_scale (float): linear scale to be applied to timesteps. Default: 1.0
        num_classes (int or None): number of classes or None if the block doesn't use class conditioning. Default: None
        activation (function): activation function. Default: torch.nn.functional.relu
        dropout (float): dropout rate at the end of each residual block
        attention_resolutions (tuple): list of relative resolutions at which to apply attention. Default: ()
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
        initial_pad (int): initial padding applied to image. Should be used if height or width is not a power of 2. Default: 0
    """

    def __init__(
            self,
            img_channels,
            base_channels,
            channel_mults=(1, 2, 4, 8),
            num_res_blocks=2,
            time_emb_dim=None,  # 时间步的嵌入向量会被映射到一个ema_dim维的空间
            time_emb_scale=1.0,
            num_classes=None,
            activation=F.relu,
            dropout=0.1,
            attention_resolutions=(),
            norm="gn",
            num_groups=32,
            initial_pad=0,
            device = None
    ):
        super().__init__()

        self.activation = activation
        self.initial_pad = initial_pad

        self.num_classes = num_classes
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(base_channels, time_emb_scale),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        ) if time_emb_dim is not None else None

        self.init_conv = nn.Conv3d(img_channels, base_channels, 3, padding=1)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = [base_channels]
        now_channels = base_channels



        #构造下采样
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(
                    now_channels,
                    out_channels,
                    dropout,
                    time_emb_dim=time_emb_dim,
                    num_classes=num_classes,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=i in attention_resolutions,
                    device=device
                ))
                now_channels = out_channels
                channels.append(now_channels)

            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))
                channels.append(now_channels)
        #中间层
        self.mid = nn.ModuleList([
            ResidualBlock(
                now_channels,
                now_channels,
                dropout,
                time_emb_dim=time_emb_dim,
                num_classes=num_classes,
                activation=activation,
                norm=norm,
                num_groups=num_groups,
                use_attention=True,
                device=None
            ),
            ResidualBlock(
                now_channels,
                now_channels,
                dropout,
                time_emb_dim=time_emb_dim,
                num_classes=num_classes,
                activation=activation,
                norm=norm,
                num_groups=num_groups,
                use_attention=False,
                device=None
            ),
        ])
        #上采样层
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks + 1):
                self.ups.append(ResidualBlock(
                    channels.pop() + now_channels,
                    out_channels,
                    dropout,
                    time_emb_dim=time_emb_dim,
                    num_classes=num_classes,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=i in attention_resolutions,
                    device=None
                ))
                now_channels = out_channels

            if i != 0:
                self.ups.append(Upsample(now_channels))

        assert len(channels) == 0

        self.out_norm = get_norm(norm, base_channels, num_groups)

        self.out_conv = nn.Conv3d(base_channels, img_channels, 3, padding=1)

    def forward(self, x, time=None, y=None,prompt=None):
        ip = self.initial_pad
        if ip != 0:
            x = F.pad(x, (ip,) * 5)

        if self.time_mlp is not None:
            if time is None:
                raise ValueError("time conditioning was specified but tim is not passed")

            time_emb = self.time_mlp(time)
        else:
            time_emb = None

        if self.num_classes is not None and y is None:
            raise ValueError("class conditioning was specified but y is not passed")

        x = self.init_conv(x)

        skips = [x]
        #仅仅在residual block中使用attn的时候需要prompt
        for layer in self.downs:
            #原始数据是39 -> 20 -> 10 -> 5
            if isinstance(layer, ResidualBlock):
                x = layer(x, time_emb, y, prompt=prompt)
            else: x = layer(x, time_emb, y)
            skips.append(x)

        for layer in self.mid:
            x = layer(x, time_emb, y,prompt=prompt)

        for layer in self.ups:
            #上采样一直到x:[1,256,40,64,64]
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x, time_emb, y)

        x = self.activation(self.out_norm(x))
        x = self.out_conv(x)

        if self.initial_pad != 0:
            return x[:, :, ip:-ip, ip:-ip,ip:-ip]
        else:
            return x