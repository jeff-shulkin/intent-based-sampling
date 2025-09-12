'''
This file serves as a dumping ground for all standard video metric calculations.
'''
from typing import Callable, Sequence, Union

import torch

from ignite.metrics import Metric, PSNR, SSIM, MeanSquaredError
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from ignite.exceptions import NotComputableError

from lpips import LPIPS

class Ignite_LPIPS(Metric):
    _state_dict_all_req_keys = ("_sum_of_batchwise_lpips", "_num_examples")

    def __init__(
        self, 
        output_transform: Callable =lambda x: x,
        device: Union[str, torch.device] = torch.device("cuda"),
        skip_unrolling: bool = False
    ):
        super(Ignite_LPIPS, self).__init__(output_transform=output_transform, device=device, skip_unrolling=skip_unrolling)
        self.lpips_model = LPIPS(net="alex").to(device)

    def _check_shape_dtype(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output
        if y_pred.dtype != y.dtype:
            raise TypeError(
                f"Expected y_pred and y to have the same data type. Got y_pred: {y_pred.dtype} and y: {y.dtype}."
            )
        
        if y_pred.shape != y.shape:
            raise ValueError(
                f"Expected y_pred and y to have the same shape. Got y_pred: {y_pred.shape} and y: {y.shape}."
            )
        
    @reinit__is_reduced
    def reset(self):
        self._sum_of_batchwise_lpips = torch.tensor(0.0, dtype=self._double_dtype, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._check_shape_dtype(output)
        y_pred, y = output[0].detach(), output[1].detach()

        self._sum_of_batchwise_lpips = torch.sum(self.lpips_model(y_pred, y))
        self._num_examples += y.shape[0]

    @sync_all_reduce("_sum_of_batchwise_lpips", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError("PSNR must have at least one example before it can be computed.")
        return (self._sum_of_batchwise_lpips / self._num_examples).item()

class VideoMetrics:
    def __init__(self, device=torch.device('cuda'), data_range=1.0) -> None:
        self.psnr_metric = PSNR(data_range=data_range, device=device)
        self.ssim_metric = SSIM(data_range=data_range, device=device)
        self.lpips_metric = Ignite_LPIPS(device=device)
        self.mse_metric = MeanSquaredError(device=device)

    def reset(self):
        self.psnr_metric.reset()
        self.ssim_metric.reset()
        self.lpips_metric.reset()
        self.mse_metric.reset()
        
    def update(self, predicted_frame, gt_frame) -> None:
        if gt_frame.dtype != predicted_frame.dtype:
            predicted_frame = predicted_frame.to(gt_frame.dtype)

        self.psnr_metric.update((predicted_frame, gt_frame))
        self.ssim_metric.update((predicted_frame, gt_frame))
        self.lpips_metric.update((predicted_frame, gt_frame))
        self.mse_metric.update((predicted_frame, gt_frame))

    def compute(self) -> None:
        self.psnr_metric.compute()
        self.ssim_metric.compute()
        self.lpips_metric.compute()
        self.mse_metric.compute()