import os
import random
import shutil

import cv2
# import lpips
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import torchvision.datasets as dset
from scipy.stats import entropy

transforms_ = [
    transforms.Resize((256, 256), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
]
transform = transforms.Compose(transforms_)

def fid(real, fake):
    print('Calculating FID...')
    print('real dir: {}'.format(real))
    print('fake dir: {}'.format(fake))
    
    #command = 'python -m pytorch_fid {} {} --gpu {}'.format(real, fake, gpu)
    command = 'python -m pytorch_fid {} {} --batch-size 1'.format(real, fake)
    os.system(command)


def LPIPS(root, real_img):
    print('Calculating LPIPS...')
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    model = loss_fn_vgg
    model.cuda()

    files = os.listdir(root)
    data = []
    for file in tqdm(files, desc='loading data'):
        img = lpips.im2tensor((lpips.load_image(os.path.join(root, file))))
        data.append(img)

    real_img = lpips.im2tensor((lpips.load_image(real_img)))

    temp = []
    for i in tqdm(range(len(data)), desc='calculating lpips'):
        d = model(data[i].cuda(), real_img.cuda(), normalize=True)
        temp.append(d.detach().cpu().numpy())

    print(np.mean(temp))

def LPIPS_normal(root, real_img):
    print('Calculating LPIPS_normal...')
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    model = loss_fn_vgg
    model.cuda()

    files = os.listdir(root)
    data = []
    for file in tqdm(files, desc='loading data'):
        img = lpips.im2tensor((lpips.load_image(os.path.join(root, file))))
        data.append(img)

    real_img = lpips.im2tensor((lpips.load_image(real_img)))

    temp = []
    for i in tqdm(range(0, len(data)-1, 1), desc='calculating lpips'):

        for j in range(i+1, len(data), 1):
            d = model(data[i].cuda(), data[j].cuda(), normalize=True)
            temp.append(d.detach().cpu().numpy())
        

    print(np.mean(temp))

def inception_score(img_dir, batch_size=100, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    dat = []
    N = len(os.listdir(img_dir))
    print('%d images to be evaluated' % N)
    for i in tqdm(range(N), desc='loading image'):
        img = Image.open(img_dir +'/'+str(i).zfill(6) +'.png').convert('RGB') 

        # img = np.array(Image.open(img_dir +'/'+str(i).zfill(6) +'.png').convert('RGB') )
        img = transform(img)
        img = img.unsqueeze(0)

        # img = torch.from_numpy(img)
        # print(img.size())
        # img = img.permute(2, 0, 1).unsqueeze(0)
        # img = 2 * img / 255.0 - 1
        # print(img.size())
        dat.append(img)
    
    dat = torch.cat(dat, dim=0)

    assert batch_size > 0
    assert N >= batch_size
    
    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).cuda()
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').cuda()
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))
    # 成batch输入inception V3计算图像所属类别
    for i in range(int(N / batch_size)):
        batch = dat[i* batch_size: (i+1) * batch_size, :, :, :].cuda()
        
        #batchv = Variable(batch), batch[100,3,256,256]
        batch_size_i = batch.size()[0]
        # 成batch计算所属类别
        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batch)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

    


parser = argparse.ArgumentParser()

parser.add_argument('--real_img', type=str, default="/data/liuhaidong/model/MindTheGap-wavelet-church/style_images/Aligned/sketch.png")
parser.add_argument('--real_dir', type=str, default="dataset/RESIDE_SOTS_outdoor/test/target")
parser.add_argument('--fake_dir', type=str,default="dataset/RESIDE_SOTS_outdoor/conclusion_RESIDE_SOTS_outdoor_1219_6500")
parser.add_argument('--mode', type=str, default='fid')

args = parser.parse_args()



if __name__ == '__main__':

    real_dir = args.real_dir
    fake_dir = args.fake_dir
    print('real dir: ', real_dir)
    print('fake dir: ', fake_dir)

    #fid(real_dir, fake_dir, args.gpu)
    if args.mode == 'LPIPS':
        LPIPS(fake_dir, args.real_img)
    
    if args.mode == 'IS':
        IS_mean, IS_std = inception_score(fake_dir, batch_size=100, resize=False, splits=10)
        print('Inception score: %.2f' % IS_mean)
        
    if args.mode == 'fid':
        fid(real_dir, fake_dir)
