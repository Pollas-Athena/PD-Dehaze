import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2  # 128/2 = 64
    emb = math.log(10000) / (half_dim - 1)  # 9.2103 / 63 = 0.146195878
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)    # 变成了1*64维度的
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]  # 16的维度是16个块块的意思 乘号前面的变成了16*1的矩阵  乘号后面是1*64的矩阵
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  #两个16*64拼接在一起 得到16*128的结果
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
                
                 
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # 自注意力
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


class DiffusionPGUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        ch, out_ch, ch_mult = self.config.model.ch, self.config.model.out_ch, tuple(self.config.model.ch_mult)  # 128 3 [1,2,3,4]转元组
        num_res_blocks = self.config.model.num_res_blocks  # 2
        attn_resolutions = self.config.model.attn_resolutions  # [16,]
        dropout = self.config.model.dropout  # 0.0
        in_channels = self.config.model.in_channels * 2 if self.config.data.conditional else self.config.model.in_channels  # 有条件就是6 没条件就是3
        resolution = self.config.data.image_size  # 64
        resamp_with_conv = self.config.model.resamp_with_conv  # True
        stage = self.config.model.stage

        self.adjust_channels = None
        self.ch = ch  # 128
        self.temb_ch = self.ch * 4  # 128 * 4 = 512
        self.ch_mult = ch_mult
        self.num_resolutions = len(ch_mult) #len(1,2,3,4) = 4
        self.num_res_blocks = num_res_blocks  # 2
        self.resolution = resolution  # 64
        self.in_channels = in_channels  # 无条件的时候是3 有条件的时候是6
        self.stage = stage

        # timestep embedding时间嵌入模块
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),  # 128 -> 512
            torch.nn.Linear(self.temb_ch, self.temb_ch),  # 512 -> 512
        ])

        # 下采样
        self.conv_in = torch.nn.Conv2d(in_channels, # 6
                                       self.ch, # 128 输出的通道
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution  # 块块大小
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(min(stage + 1, self.num_resolutions)):   # 4

            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]  #128*1  1是[11234]中的1
            block_out = ch*ch_mult[i_level]    #128*1  1是[1234]中的1
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
              
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
        

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # 上采样
        self.up = nn.ModuleList()
        for i_level in reversed(range(min(stage + 1, self.num_resolutions))):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t, stage=0):
        # timestep embedding
        print(f"Input shape: {x.shape}")
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        hs = [self.conv_in(x)]
        print(f"After conv_in: {self.conv_in(x).shape}") 
        
        # 根据当前训练阶段选择适当的分辨率模块
        for i_level in range(min(stage + 1, self.num_resolutions)):  # stage 控制当前训练阶段
            for i_block in range(self.num_res_blocks):
                # import pdb
                # pdb.set_trace()
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                print(f"After down block {i_block}: {h.shape}")
                
                hs.append(h)
            if i_level != min(stage + 1, self.num_resolutions) - 1:
                h = self.down[i_level].downsample(hs[-1])
                hs.append(h)

                print(f"After downsample {i_level}: {h.shape}")


        # 中间模块
        h = hs[-1]
      
        # # 动态调整通道数
        # if h.size(1) != 512:  # 检查通道数是否为 512

        #     if self.adjust_channels is None:
        #         # 定义动态调整层：输入通道数 -> 512
        #         self.adjust_channels = nn.Conv2d(h.size(1), 512, kernel_size=1)
        #     h = self.adjust_channels(h)  # 调整 h 的通道数

    
        h = self.mid.block_1(h, temb)
        print(f"After mid.block_1: {h.shape}")
        h = self.mid.attn_1(h)
        print(f"After mid.attn_1: {h.shape}")
        h = self.mid.block_2(h, temb)
        print(f"After mid.block_2: {h.shape}")

        # 上采样
        for i_level in reversed(range(min(stage + 1, self.num_resolutions))):
            for i_block in range(self.num_res_blocks+1):
                # import pdb
                # pdb.set_trace()
                # if h.shape[1] != hs[-1].shape[1]:
                #     h = nn.Conv2d(h.shape[1], hs[-1].shape[1], kernel_size=1).to(h.device)(h)
                h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
                print(f"After up block {i_block}: {h.shape}")
            if i_level != 0:
                h = self.up[i_level].upsample(h)
                print(f"After upsample {i_level}: {h.shape}")

        # 输出
        h = self.norm_out(h)
        print(f"After norm_out: {h.shape}")
        h = nonlinearity(h)
        print(f"After nonlinearity: {h.shape}")
        h = self.conv_out(h)
        print(f"After conv_out: {h.shape}")

        return h
# net = DiffusionUNet()

# 定义一个简单的配置对象（可以根据需要修改）
class Config:
    class Model:
        def __init__(self):
            self.in_channels = 3
            self.out_ch = 3
            self.ch = 128

            self.ch_mult = [1, 2, 3, 4]
            self.num_res_blocks = 2
            self.attn_resolutions = [16]
            self.dropout = 0.0
            
            self.resamp_with_conv = True
            self.stage = 2

    class Data:
        def __init__(self):
            self.conditional = True
            self.image_size = 8

    def __init__(self):
        self.model = self.Model()
        self.data = self.Data()

config = Config()

# 创建模型
model = DiffusionPGUNet(config)

# 创建一个随机输入图像，形状为 [batch_size, channels, height, width]
# 假设 batch_size = 1, channels = 3 (RGB图像), height = 64, width = 64
# x = torch.randn(1, 6, 8, 8)
x = torch.randn(1, 6, 64, 64)

# 创建一个时间步长参数，假设我们使用一个时间步 t =E 10
t = torch.tensor([10])

# 选择当前训练阶段，例如，stage=0Es
# stage=1

# 前向传播
output = model(x, t,stage=2)

# 检查输出的形状
print(f"Output shape: {output.shape}")
