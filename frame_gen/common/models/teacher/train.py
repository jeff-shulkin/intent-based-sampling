import argparse
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from time import time
import numpy as np
from tqdm import tqdm

from model import NextFrameTransformerTeacher
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
    metrics = VideoMetrics()
    train_loss_history = []
    train_video_metrics_history = []

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
        epoch_video_metrics_history = []

        # Create epoch-specific progress bar
        step_pbar = tqdm(dls["train_dl"], 
                         desc='Steps', 
                         unit='step',
                         total=num_steps_per_epoch,
                         position=1,    # This is key for nesting
                         leave=False)

        for ref_frame, event_voxels, gt_next_frame in step_pbar:
            # Grab past RGB frame, current event voxels, and next RGB frame
            load_start = time()
            ref_frame, event_voxels, gt_next_frame = ref_frame.to(device), event_voxels.to(device), gt_next_frame.to(device)
            load_end = time()

            # Predict the next frame based on current RGB frame and event voxels
            pred_start = None
            pred_end = None
            pred_frame = None
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred_start = time()
                pred_frame = model(ref_frame, event_voxels)
                loss = loss_function(pred_frame, gt_next_frame)
                pred_end = time()

            epoch_loss_history.append(loss.item())
            
            # Scale loss using GradScaler
            back_start = time()
            scaler.scale(loss).backward()
            back_end = time()

            # Optimize
            opt_start = time()
            scaler.step(optimizer)
            opt_end = time()
            
            # Update scaler for next iteration
            scaler.update()

            # Zero out gradients
            optimizer.zero_grad(set_to_none=True)

            # Calculate per-item video metrics during epoch
            #pred_np = pred_frame.detach().cpu().numpy()
            #gt_np = gt_next_frame.detach().cpu().numpy()

            #batch_psnr = metrics.batched_psnr(im=pred_np, gt_im=gt_np)
            #batch_ssim = metrics.batched_ssim(im=pred_np, gt_im=gt_np)
            #batch_lpips = metrics.batched_lpips(im=pred_np, gt_im=gt_np)

            #for psnr, ssim in zip(batch_psnr, batch_ssim):
            #    epoch_video_metrics_history.append((psnr, ssim, 0.0))

            step_end = time()

            print("Step timing metrics:")
            print(f"Step Time: {step_end - load_start} seconds")
            print(f"Load Time: {load_end - load_start} seconds")
            print(f"Prediction Time: {pred_end - pred_start} seconds")
            print(f"Backwards Time: {back_end - back_start}")
            print(f"Optimization Time: {opt_end - opt_start} seconds")

        # Close epoch-specific progress bar
        step_pbar.close()

        end_time = time()
        train_loss_history.append(sum(epoch_loss_history) / len(epoch_loss_history))
        train_video_metrics_history.append(tuple(np.mean(epoch_video_metrics_history, axis=0)))

        torch.cuda.empty_cache()

        print(f"Epoch: {epoch}")
        print(f"Training loss: {train_loss_history[-1]}")
        print(f"Average PSNR: {train_video_metrics_history[-1][0]}")
        print(f"Average SSIM: {train_video_metrics_history[-1][1]}")
        print(f"Average LPIPS: {train_video_metrics_history[-1][2]}")
        print(f"Training time: {end_time - start_time}")
        validate_model(model, loss_function, dls["val_dl"])

    # Close overall progress progress bar once all epochs have finished
    epoch_pbar.close()

# Validation function
def validate_model(model, loss_function, val_dl, device):
    # Variables to assess performance
    epoch_loss_history = []
    validation_loss_history = []
    validation_accuracy_history = []
    num_correct = 0
    num_total = 0

    with torch.no_grad():
        for inputs, true_labels in val_dl:
            inputs, true_labels = inputs.to(device), true_labels.to(device)
            true_labels = true_labels.squeeze(1)
            outputs = model(inputs)
            loss = loss_function(outputs, true_labels)
            epoch_loss_history.append(loss.item())
            _, predicted_labels = torch.max(outputs, 1)
            num_total += true_labels.size(0)
            num_correct += (predicted_labels == true_labels).sum().item()

    validation_loss_history.append(sum(epoch_loss_history) / len(epoch_loss_history))
    validation_accuracy_history.append(100 * num_correct / num_total)
    print(f"Validation loss: {validation_loss_history[-1]}")
    print(f"Validation accuracy: {validation_accuracy_history[-1]} %")

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
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, persistent_workers=True, drop_last=True, num_workers=args.num_workers)
    val_dl = DataLoader(val_set, batch_size=batch_size, shuffle=True, pin_memory=True, persistent_workers=True, drop_last=True, num_workers=args.num_workers)
    test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=True, pin_memory=True, persistent_workers=True, drop_last=True, num_workers=args.num_workers)
    dls = {
        "train_dl" : train_dl,
        "val_dl": val_dl,
        "test_dl": test_dl
    }

    # Initialize model, loss function, optimizer
    model = NextFrameTransformerTeacher(
        image_size=args.image_size,
        patch_size=16,
        embed_dim=512,
        nhid=2048,
        nhead=8,
        nlayers=8,
        dropout=0.1,
        num_voxels=5
    ).to(device)
    model = torch.compile(model, mode='reduce-overhead')

    learning_rate = 1e-4
    num_epochs = args.num_epochs
    loss_function = nn.L1Loss()  # TODO: Define proper loss function. Probably combination of L1Loss, LPIPS, maybe PSNR/SSIM?
    optimizer = optim.AdamW(params=model.parameters(), lr=learning_rate)

    # Train the teacher model
    print("Starting teacher model training...")
    train_model(
        model=model,
        loss_function=loss_function, 
        optimizer=optimizer, 
        dls=dls, 
        num_epochs=num_epochs, 
        device=device,
        use_amp=args.use_amp)

    # Save the teacher model
    model_filename = "teacher.pth"
    print(f"Saving model")
    torch.save(obj=model, f=model_filename)



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
    