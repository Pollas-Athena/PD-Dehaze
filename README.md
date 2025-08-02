# PD-Dehaze
for training: run train_diffusion.py
for test: run eval_diffusion.py
for metric: run calculate_FID.py calculate_LPIPS.py calculate_psnr_ssim.py to get FID, LPIPS, PSNR, SSIM values To get our metric CDiff (color difference) which is designed for Evaluating the degree of color shift, please run the for_color_shift.py
configs.yml include all you need, the default path is the author's custom, so feel free to change it.
We provide the model on the trained dataset NH-HAZE and DENSE-HAZE, in the following path:
We provide the dataset NH-HAZE and DENSE-HAZE, in the following path: (The limited upload size is not enough to fit the dataset and the model)
