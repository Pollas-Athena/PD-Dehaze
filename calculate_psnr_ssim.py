import os
import cv2
import logging
import eval_diffusion
from utils.metrics import calculate_psnr, calculate_ssim
from resizeAtoB import resize_image_to_match

# 配置日志记录器
logging.basicConfig(filename='logger_RESIZE/NH_UCL.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 创建日志记录器
logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)



config = eval_diffusion.config_get()
gt_path = os.path.join(config.data.test_data_dir, 'target/')  # 测试文件的标签路径
results_path = config.data.test_save_dir                      # 测试文件的标签路径

imgsName = sorted(os.listdir(results_path))
gtsName = sorted(os.listdir(gt_path))

cumulative_psnr, cumulative_ssim = 0, 0
for i in range(len(imgsName)):
    print('Processing image: %s' % (imgsName[i]))
    logger.info(f'Processing image: {imgsName[i]}')
    res = cv2.imread(os.path.join(results_path, imgsName[i]), cv2.IMREAD_COLOR)
    gt = cv2.imread(os.path.join(gt_path, gtsName[i]), cv2.IMREAD_COLOR)
    import pdb
    pdb.set_trace()
    # res2 = resize_image_to_match(os.path.join(results_path, imgsName[i]),os.path.join(gt_path, gtsName[i]))
    cur_psnr = calculate_psnr(res, gt, test_y_channel=True)
    cur_ssim = calculate_ssim(res, gt, test_y_channel=True)
    print('PSNR is %.4f and SSIM is %.4f' % (cur_psnr, cur_ssim))
    # 记录运行结果到日志
    logger.info(f"PSNR is: {cur_psnr}")
    logger.info(f"SSIM is: {cur_ssim}")
    cumulative_psnr += cur_psnr
    cumulative_ssim += cur_ssim
print('Testing set, PSNR is %.4f and SSIM is %.4f' % (cumulative_psnr / len(imgsName), cumulative_ssim / len(imgsName)))
print(results_path)
logger.info(f"Testing set, PSNR is: {cumulative_psnr / len(imgsName)}")
logger.info(f"Testing set, SSIM is: {cumulative_ssim / len(imgsName)}")
logger.info(f"Testing root: {results_path}")

