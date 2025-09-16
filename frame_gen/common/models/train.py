import argparse
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from time import time
import numpy as np
from tqdm import tqdm

from frame_gen.common.models.model import FusionFrameGen
from frame_gen.common.metrics.loss_fn import CompositeLoss

from frame_gen.datasets.HS_ERGB_dataset import HSERGBDataset
from frame_gen.datasets.BS_ERGB_dataset import BSERGBDataset
# TODO: Implement MVSEC Pytorch Dataset
#from frame_gen.datasets.MVSEC_dataset import MVSECDataset

from frame_gen.common.metrics.video_metrics import VideoMetrics

from tools.pytorch_tools import determine_device, split_dataset
from tools.os_tools import image_size_arg

# Training function
def train_model(model, loss_function, optimizer, dls: list[DataLoader], num_epochs: int, device, use_amp: bool):
    # Initialize video metrics and training histories
    metrics = VideoMetrics(device=device)
    train_loss_history = []

    # Initialize scaler
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # Create overall training progress bar
    num_steps_per_epoch = len(dls["train_dl"])
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", unit='epoch', position=0)

    for epoch in epoch_pbar:
        start_time = time()

        # =====================================================================
        #  TRAIN
        # =====================================================================

        model.train() # Set model to train
        epoch_loss_history = []

        # Create epoch-specific progress bar
        step_pbar = tqdm(dls["train_dl"], 
                         desc='Steps', 
                         unit='step',
                         total=num_steps_per_epoch,
                         position=1,
                         leave=False)
        
        # Reset internal video metrics
        metrics.reset()

        for ref_frame, event_voxels, gt_next_frame in step_pbar:
            # Grab past RGB frame, current event voxels, and next RGB frame
            ref_frame, event_voxels, gt_next_frame = ref_frame.to(device), event_voxels.to(device), gt_next_frame.to(device)

            # Predict the next frame based on current RGB frame and event voxels
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred_frame = model(ref_frame, event_voxels)
                loss = loss_function(pred_frame, gt_next_frame)

            epoch_loss_history.append(loss.item())
            
            # Scale loss using GradScaler
            scaler.scale(loss).backward()

            # Optimize
            scaler.step(optimizer)
            
            # Update scaler for next iteration
            scaler.update()

            # Zero out gradients
            optimizer.zero_grad(set_to_none=True)

            # Calculate per-item video metrics during epoch
            metrics.update(gt_frame=gt_next_frame, predicted_frame=pred_frame)

        # Close epoch-specific progress bar
        step_pbar.close()

        end_time = time()

        # Compute epoch-specific video metrics
        epoch_metrics = metrics.compute()
        train_loss_history.append(sum(epoch_loss_history) / len(epoch_loss_history))

        torch.cuda.empty_cache()

        print(f"Epoch: {epoch}")
        print(f"Training loss: {train_loss_history[-1]}")
        print(f"Training time: {end_time - start_time}")
        
        print(f"Training PSNR: {epoch_metrics["psnr"]}")
        print(f"Training SSIM: {epoch_metrics["ssim"]}")
        print(f"Training LPIPS: {epoch_metrics["lpips"]}")
        print(f"Training MSE: {epoch_metrics["mse"]}")

        validate_model(model, loss_function, dls["val_dl"])

    # Close overall progress progress bar once all epochs have finished
    epoch_pbar.close()

# Validation function
def validate_model(model, loss_function, val_dl, device, use_amp: bool = True):
    model.eval()
    metrics = VideoMetrics(device=device)
    epoch_loss_history = []

    with torch.no_grad():
        val_pbar = tqdm(val_dl, desc="Validating", unit="batch", leave=False)
        
        for ref_frame, event_voxels, gt_next_frame in val_pbar:
            ref_frame = ref_frame.to(device)
            event_voxels = event_voxels.to(device)
            gt_next_frame = gt_next_frame.to(device)

            # Forward pass
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred_frame = model(ref_frame, event_voxels)
                loss = loss_function(pred_frame, gt_next_frame)

            epoch_loss_history.append(loss.item())

            # Update video metrics
            metrics.update(predicted_frame=pred_frame, gt_frame=gt_next_frame)

        val_pbar.close()

    # Compute metrics
    epoch_metrics = metrics.compute()
    val_loss = sum(epoch_loss_history) / len(epoch_loss_history)

    print(f"\nValidation results:")
    print(f"Loss: {val_loss:.6f}")
    print(f"PSNR: {epoch_metrics['psnr']:.4f}")
    print(f"SSIM: {epoch_metrics['ssim']:.4f}")
    print(f"LPIPS: {epoch_metrics['lpips']:.4f}")
    print(f"MSE: {epoch_metrics['mse']:.6f}")

    return {
        "loss": val_loss,
        **epoch_metrics
    }

def train_teacher(args):
    device = determine_device()

    # Load relevant datasets
    print("Loading datasets...")
    HSERGB = HSERGBDataset(pathlib.Path(args.hs_ergb), image_size=args.image_size)
    #BSERGB = BSERGBDataset(pathlib.Path(args.bs_ergb))
    #MVSEC = MVSECDataset(pathlib.Path(args.mvsec))
    print("All datasets loaded.")

    # Concatenate all datasets
    dataset = ConcatDataset([HSERGB])
    print(f"Number of samples: {len(dataset)}")

    # Split concatenated dataset: 80% training, 10% validation, 10% test
    train_set, val_set, test_set = split_dataset(dataset, ptrain=0.8, pval=0.1)

    # Create DataLoaders
    batch_size = args.batch_size
    train_dl = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True, 
        persistent_workers=True, 
        drop_last=True, 
        num_workers=args.num_workers
    )
    val_dl = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        num_workers=args.num_workers
    )
    test_dl = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        num_workers=args.num_workers
    )
    dls = {
        "train_dl" : train_dl,
        "val_dl": val_dl,
        "test_dl": test_dl
    }

    # Initialize model, loss function, optimizer
    model = FusionFrameGen(
        image_size=args.image_size,
        patch_size=16,
        embed_dim=512,
        nhid=2048,
        nhead=8,
        nlayers=8,
        dropout=0.1,
        num_voxels=5
    ).to(device)
    
    compiled_model = torch.compile(model, mode='reduce-overhead')

    learning_rate = 1e-4
    num_epochs = args.num_epochs
    loss_fn_dict = {
        "L1": (nn.L1Loss(), 1.0),
    }
    loss_function = CompositeLoss(loss_fn_dict)
    optimizer = optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # Train the teacher model
    print("Starting teacher model training...")
    train_model(
        model=compiled_model,
        loss_function=loss_function, 
        optimizer=optimizer, 
        dls=dls, 
        num_epochs=num_epochs, 
        device=device,
        use_amp=args.use_amp
    )

    # Save the teacher model
    model_filename = "model.pth"
    print(f"Saving model...")
    torch.save(obj=model.state_dict(), f=model_filename)
    print(f"Model saved.")



if __name__=="__main__":
    # Dataset arguments
    parser = argparse.ArgumentParser(description="Training script for student transformer model.")
    parser.add_argument("--hs_ergb", type=str, default="../frame_gen/datasets/hs-ergb-dataset")
    parser.add_argument("--bs_ergb", type=str, default="../frame_gen/datasets/bs-ergb-dataset")
    #parser.add_argument("--mvsec", type=str, default="../frame_gen/datasets/mvsec-dataset")
    
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--image_size", type=image_size_arg, default=(224,224))

    # Model training parameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--use_amp", type=bool, default=True)
    
    args = parser.parse_args()

    train_teacher(args=args)
    