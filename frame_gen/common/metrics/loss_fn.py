import torch
import torch.nn as nn

from tools.pytorch_tools import _lpips_scale

class CompositeLoss(nn.Module):
    def __init__(self, loss_fn_dict, device):
        super(CompositeLoss, self).__init__()
        self.loss_fns = loss_fn_dict

    def forward(self, pred_im, gt_im):
        total_loss = 0.0
        for name, (loss_fn, weight) in self.loss_fns.items():
            curr_loss = 0.0

            if name.lower() == "lpips":
                # Scale LPIPS to [-1, 1] for proper calculation
                pred_scaled, gt_scaled = _lpips_scale(pred_im), _lpips_scale(gt_im)
                curr_loss = loss_fn(pred_scaled, gt_scaled).mean()
            
            if name.lower() == "ssim":
                curr_loss = 1 - loss_fn(pred_im, gt_im)
            
            else:
                curr_loss = loss_fn(pred_im, gt_im)
            
            total_loss += weight * curr_loss

        return total_loss
