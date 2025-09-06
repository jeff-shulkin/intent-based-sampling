'''
This file serves as a dumping ground for all standard video metric calculations.
'''
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from lpips import LPIPS
import torch

class VideoMetrics:
    def __init__(self, batch_size=16):
        self.lpips = LPIPS(net="alex")
        self.lpips_device = next(self.lpips.parameters()).device
        self.batch_size = batch_size

    def batched_psnr(self, gt_im=None, im=None, data_range=255) -> list[float]:
        psnrs = []
        for sample in range(self.batch_size):
            curr_gt_im = gt_im[sample].transpose(1, 2, 0)
            curr_pred_im = im[sample].transpose(1, 2, 0)
            psnrs.append(peak_signal_noise_ratio(image_true=curr_gt_im, image_test=curr_pred_im, data_range=data_range))

        return psnrs
    
    def batched_ssim(self, gt_im=None, im=None, data_range=255) -> list[float]:
        ssims = []
        for sample in range(self.batch_size):
            curr_gt_im = gt_im[sample].transpose(1, 2, 0)
            curr_pred_im = im[sample].transpose(1, 2, 0)
            ssims.append(structural_similarity(im1=curr_gt_im, im2=curr_pred_im, data_range=data_range, channel_axis=2))

        return ssims
    
    def batched_lpips(self, gt_im=None, im=None) -> list[float]:
        curr_gt_im_tensor = torch.from_numpy(gt_im).float().to(self.lpips_device)
        curr_pred_im_tensor = torch.from_numpy(im).float().to(self.lpips_device)

        if curr_pred_im_tensor.max() <= 1.0 or curr_gt_im_tensor.max() <= 1.0:
            curr_pred_im_tensor = (curr_pred_im_tensor * 2) - 1
            curr_gt_im_tensor = (curr_gt_im_tensor * 2) - 1
        
        lpips = []
        with torch.no_grad():
            lpips = self.lpips(in0=curr_gt_im_tensor, in1=curr_pred_im_tensor).cpu().numpy().tolist()

        return lpips
    
    def mse(self, gt_im, im) -> list[float]:
        return mean_squared_error(im, gt_im)
    
    def vmaf(self, im) -> float:
        pass