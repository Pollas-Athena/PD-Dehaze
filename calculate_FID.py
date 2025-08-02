# import torch
# import torchvision
# import torchvision.transforms as transforms
# from pytorch_fid import fid_score
# import os
# 无日志版本
# def fid(real, fake):
#     print('Calculating FID...')
#     print('real dir: {}'.format(real))
#     print('fake dir: {}'.format(fake))
#     #command = 'python -m pytorch_fid {} {} --gpu {}'.format(real, fake, gpu)
#     command = 'python -m pytorch_fid {} {} --batch-size 1'.format(real, fake)
#     os.system(command)

# if __name__ == '__main__':

#     real_dir = 'dataset/RESIDE_SOTS_outdoor/test/target'
#     fake_dir = 'dataset/RESIDE_SOTS_outdoor/conclusion_RESIDE_SOTS_outdoor_1219_6500'
#     print('real dir: ', real_dir)
#     print('fake dir: ', fake_dir)
        
#     fid(real_dir, fake_dir)

import os
import random
import shutil
import eval_diffusion
import cv2
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
import os
import subprocess
import logging

def setup_logging(log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    return logger

def fid(real, fake, logger_fid):
    print('Calculating FID...')
    print('real dir: {}'.format(real))
    print('fake dir: {}'.format(fake))
    #command = 'python -m pytorch_fid {} {} --gpu {}'.format(real, fake, gpu)
    command = 'python -m pytorch_fid {} {} --batch-size 1'.format(real, fake)
    # os.system(command)
    try:
        # 使用 subprocess 模块运行命令并捕获输出
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout
        error = result.stderr

        # 将输出记录到日志中
        logger_fid.info(f"Command: {command}")
        logger_fid.info(f"Output: {output}")
        logger_fid.info(f"Error: {error}")
        
        print('FID calculation successful.')

    except subprocess.CalledProcessError as e:
        # 如果命令执行失败，记录错误信息到日志
        logger_fid.error(f"Command failed with error: {e}")
        print('FID calculation failed.')    

if __name__ == '__main__':
    # 设置日志文件路径
    log_file_path_fid = 'logger/fid_conclusion_RE_ucl.txt'
    # 检查文件路径是否存在，如果不存在则创建
    if not os.path.exists(os.path.dirname(log_file_path_fid)):
        os.makedirs(os.path.dirname(log_file_path_fid))
    setup_logging(log_file_path_fid)
    
    logger_fid = setup_logging(log_file_path_fid)
    
    # real_dir = 'dataset/RESIDE_SOTS_outdoor/test_tr/target'
    # fake_dir = 'dataset/RESIDE_SOTS_outdoor/conclusion_RE_6500_tr'
    
    config = eval_diffusion.config_get()
    real_dir = os.path.join(config.data.test_data_dir, 'target/')  # 测试文件的标签路径
    fake_dir = config.data.test_save_dir                      # 测试文件的标签路径

    
    print('real dir: ', real_dir)
    print('fake dir: ', fake_dir)
        
    fid(real_dir, fake_dir, logger_fid)