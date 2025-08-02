import cv2
import numpy as np
import os


def mse_hsv_images(folder1, folder2):
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)
    mse_values = []
    for filename1, filename2 in zip(files1, files2):
        # 读取图像
        img1 = cv2.imread(os.path.join(folder1, filename1))
        img2 = cv2.imread(os.path.join(folder2, filename2))
        hsv_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        # 计算均方差
        mse = np.mean((hsv_img1 - hsv_img2) ** 2)
        mse_values.append(mse)   
    return mse_values

mse_gt_one = mse_hsv_images('gt', 'one')
mse_gt_two = mse_hsv_images('gt', 'two')
mse_gt_our = mse_hsv_images('gt', 'our')

print("mse_gt_one:", mse_gt_one)
print("mse_gt_one:", np.mean(mse_gt_one))
print("mse_gt_two:", mse_gt_two)
print("mse_gt_one:", np.mean(mse_gt_two))
print("mse_gt_our:", mse_gt_our)
print("mse_gt_one:", np.mean(mse_gt_our))













# mse_one_gt = mean_squared_error(hsv_image_gt, hsv_image_one)
# mse_two_gt = mean_squared_error(hsv_image_gt, hsv_image_two)
# mse_our_gt = mean_squared_error(hsv_image_gt, hsv_image_our)

# import pdb
# pdb.set_trace()

# image_path = 'one/55_hazy.png'
# image = cv2.imread(image_path)
# resized_image = cv2.resize(image, (1600, 1200))

# cv2.imwrite(image_path, resized_image)
