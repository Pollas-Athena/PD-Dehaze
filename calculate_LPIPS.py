import lpips
from torchvision.transforms import ToTensor, Resize
from PIL import Image
import os
import logging
from tqdm import tqdm
import eval_diffusion

def setup_logging(log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def calculate_lpips(model, haze_dataset_path, dehaze_dataset_path):
    lpips_scores = []
    transform = ToTensor()
    
    haze_images = sorted(os.listdir(haze_dataset_path))
    dehaze_images = sorted(os.listdir(dehaze_dataset_path))  
     
    # Calculate the total number of iterations
    total_iterations = min(len(os.listdir(haze_dataset_path)), len(os.listdir(dehaze_dataset_path)))

    with tqdm(total=total_iterations, desc="Processing") as pbar:
        for haze_filename, dehaze_filename in zip(haze_images, dehaze_images):
            if (haze_filename.endswith(".png") or haze_filename.endswith(".jpg") or haze_filename.endswith(".bmp")) \
                    and (dehaze_filename.endswith(".png") or dehaze_filename.endswith(".jpg") or dehaze_filename.endswith(".bmp")):

                haze_image_path = os.path.join(haze_dataset_path, haze_filename)
                dehaze_image_path = os.path.join(dehaze_dataset_path, dehaze_filename)

                # 通过添加 Resize 操作确保输入图像的大小一致
                haze_image = transform(Resize((256, 256))(Image.open(haze_image_path))).unsqueeze(0)
                dehaze_image = transform(Resize((256, 256))(Image.open(dehaze_image_path))).unsqueeze(0)

                lpips_score = model(haze_image, dehaze_image).item()
                lpips_scores.append(lpips_score)

                # 记录到日志文件
                logging.info(f"{haze_filename},{dehaze_filename},{lpips_score}")
                pbar.update(1)
    return lpips_scores

# 设置日志文件路径
log_file_path = 'logger/lpips_conclusion_RE_UCL.txt'
# 检查文件路径是否存在，如果不存在则创建
if not os.path.exists(os.path.dirname(log_file_path)):
    os.makedirs(os.path.dirname(log_file_path))
setup_logging(log_file_path)

# Load LPIPS model
lpips_model = lpips.LPIPS(net='alex', verbose=1)

# Replace with your haze and dehaze dataset paths
# haze_dataset_path = 'dataset/RESIDE_SOTS_outdoor/test/target'
# dehaze_dataset_path = 'dataset/RESIDE_SOTS_outdoor/conclusion_RESIDE_SOTS_outdoor_1219_6500'

config = eval_diffusion.config_get()
haze_dataset_path = os.path.join(config.data.test_data_dir, 'target/')  # 测试文件的标签路径
dehaze_dataset_path = config.data.test_save_dir  

# Calculate LPIPS scores for the datasets
lpips_scores = calculate_lpips(lpips_model, haze_dataset_path, dehaze_dataset_path)

# Calculate average LPIPS score
average_lpips = sum(lpips_scores) / len(lpips_scores)

print(f"Average LPIPS Score for Haze and Dehaze Dataset: {average_lpips}")
logging.info(f"Average LPIPS Score for Haze and Dehaze Dataset: {average_lpips}")



