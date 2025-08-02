import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import utils
import torchvision
from models.unet import DiffusionUNet
 

def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)  # 正则化
    

class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data.to('cuda')


    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

# 我知道这个是干啥的了，突然我就顿悟了 就是改beta大小的，我太特么聪明了
def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

# 这里面的第一个参数是扩散的unet模型 x是64*6*64*64的输入和标签，t是64的张量，e是给标签添加了噪声的输入，b是2000个beta
def noise_estimation_loss(model, x0, t, e, b):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)  # 到index_select是从累乘的张量中选取第t个元素，得到一个新的张量。
    # 我怎么感觉这个噪声它没加进去呢?
    # x.shape = 16*3*64*64 是标签的16个块块
    x = x0[:, 3:, :, :] * a.sqrt() + e * (1.0 - a).sqrt()  # 这里是x_t=系数x_0+系数z  x_0直接就是6通道,我不理解 理解了 因为后面三通道是标签
    output = model(torch.cat([x0[:, :3, :, :], x], dim=1), t.float()) # ||𝜖−ϵ_θ (√(¯(𝛼_t )) x_0+√({1−¯(𝛼_t )}ϵ),t)|| 这里的output是ϵ_θ

    toPIL = torchvision.transforms.ToPILImage() #这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    pic_e = toPIL(e[0])
    pic2 = toPIL(x0[:,:3,:,:][0])
    pic3 = toPIL(x[0])
    
    pic_e.save('train_e0.jpg')
    pic2.save('train_xwu.jpg')
    pic3.save('train_x1.jpg')

    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

# def noise_estimation_loss(model, x0, t, e, b):
#     a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)  # 到index_select是从累乘的张量中选取第t个元素，得到一个新的张量。
#     # 我怎么感觉这个噪声它没加进去呢?
#     x = x0[:, 3:, :, :] * a.sqrt() + e * (1.0 - a).sqrt()  # 这里是x_t=系数x_0+系数z  
#     print("0000000000000000000000000")  
#     print(x0.shape)  ## x0.shape=torch.Size([16, 6, 64, 64])
#     print(x.shape)   ## x.shape=torch.Size([16, 3, 64, 64])
#     # 这里取得的是前三个通道进行拼接,所以正确的写法应该是x0[:,:3,:,:],这里表示的是取第二个维度上的前三个元素与进行拼接.
#     # x_pinjie = torch.cat([x0[:, :3, :, :], x], dim=1)
#     # print("11111111111111111111111111111")
#     # print(x_pinjie.shape)  ## torch.Size([16, 6, 64, 64])
#     output = model(torch.cat([x0[:, :3, :, :], x], dim=1), t.float()) # ||𝜖−ϵ_θ (√(¯(𝛼_t )) x_0+√({1−¯(𝛼_t )}ϵ),t)|| 这里的output是ϵ_θ
#     return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

# 扩散模型加噪+训练过程
class DenoisingDiffusion(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device

        self.model = DiffusionUNet(config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)

        self.num_timesteps = betas.shape[0]  # = 2000

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])

        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        if os.path.isfile(self.config.training.resume):
            self.load_ddm_ckpt(self.config.training.resume)

        # 训练
        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('=> current epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                # x.shape = torch.Size([4, 16, 6, 64, 64])
                # 其中后面的[16，6，64，64] 就是mydataset经过get_images后的16个块块，输入和标签共计6通道，64*64的裁切块
                # 前面的数字是training.batch_size

                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                # x.shape() torch.Size([64, 6, 64, 64]) 前两个维度给拼接在一起
                n = x.size(0)   
                # n = 64
                data_time += time.time() - data_start
                self.model.train()  # 去到unet里面了
                self.step += 1
                # import pdb
                # pdb.set_trace()
                x = x.to(self.device)
                x = data_transform(x)
               
                e = torch.randn_like(x[:, 3:, :, :]) # e这里是添加了噪声的x  是在标签上添加均值为0，方差为1的正态分布的随机值的噪声
                from torchvision import transforms
 
                # toPIL = transforms.ToPILImage() #这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
                # # img = torch.randn(3,128,64)
                # pic = toPIL(e[0])
                # pic2 = toPIL(x[:,:3,:,:][0])
                # pic3 = toPIL(x[:,3:,:,:][0])
                # pic.save('e0.jpg')
                # pic2.save('x0.jpg')
                # pic3.save('x_y0.jpg')

                b = self.betas # tensor([0.0100,0.0100...0.0002,0.0002])

                # antithetic sampling 特么的在这噪声估计 老子找了半天
                
                # 在0到最大的时间步2000，任选了一个t，这里还得看看，具体t是怎么生成的，和batch size一样大小的t,t和batchsize的维度是一样的，因为每张图像都有一个t?
                # high= 2000，size = 64/2+1 =33 
                # t = 在0-2000中随机生成整数的33大小的张量
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
              
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]# 取前n个 也就是前64个 64=batch_size*一张图分成的16个块块
                   
                loss = noise_estimation_loss(self.model, x, t, e, b)
                # 这里面的第一个参数是扩散的unet模型 x是64*6*64*64的输入和标签，t是64的张量，e是给标签添加了噪声的输入，b是2000个beta

                if self.step % 10 == 0:
                    print('step: %d, loss: %.6f, time consumption: %.6f' % (self.step, loss.item(), data_time / (i+1)))

                # 更新参数
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step)

                # 保存模型
                if epoch % 10 == 0 or self.step == 1:
                # if self.step % self.config.training.snapshot_freq == 0 or self.step == 1:
                    utils.logging.save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'config': self.config
                    # }, filename=self.config.training.resume)
                    }, filename=self.config.training.resume + str(epoch))

    def sample_image(self, x_cond, x, last=True, patch_locs=None, patch_size=None):
        skip = self.config.diffusion.num_diffusion_timesteps // self.config.sampling.sampling_timesteps  #2000/25 = 80
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        if patch_locs is not None:
            xs = utils.sampling.generalized_steps_overlapping(x, x_cond, seq, self.model, self.betas, eta=0.,
                                                              corners=patch_locs, p_size=patch_size)
        else:
            xs = utils.sampling.generalized_steps(x, x_cond, seq, self.model, self.betas, eta=0.)
        if last:
            xs = xs[0][-1]
        return xs
    
    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.config.data.val_save_dir, str(self.config.data.image_size))
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                break
            n = x.size(0)
            x_cond = x[:, :3, :, :].to(self.device)  # 条件图像：有雾图像
            x_cond = data_transform(x_cond)
            x = torch.randn(n, 3, self.config.data.image_size, self.config.data.image_size, device=self.device)
            x = self.sample_image(x_cond, x)
            x = inverse_data_transform(x)
            x_cond = inverse_data_transform(x_cond)

            for i in range(n):
                utils.logging.save_image(x_cond[i], os.path.join(image_folder, str(step), f"{i}_cond.png"))
                utils.logging.save_image(x[i], os.path.join(image_folder, str(step), f"{i}.png"))
