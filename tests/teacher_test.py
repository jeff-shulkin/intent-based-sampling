import numpy as np
import time
import torch
from torch.utils.data import DataLoader

from frame_gen.datasets.HS_ERGB_dataset import HSERGBDataset
from tools.pytorch_tools import determine_device, split_dataset

import pathlib
import argparse

# Function to set up model for inference testing
def model_setup(model_weights_path: pathlib.Path):
    # Resolve model path
    model_weights_path = model_weights_path.expanduser().resolve()

    # Load model
    
    model = torch.load(model_weights_path, weights_only=False)

    # Compile model for faster inference time
    torch.compile(model, mode='reduce-overhead')

    # Load model onto GPU
    device = determine_device()
    model.to(device)

    # Set model to evaluation mode
    model.eval()

    return model, device

def load_test_dataset(hs_ergb_path: pathlib.Path):
    HSERGB = HSERGBDataset(hs_ergb_path)

    # Split concatenated dataset: 80% training, 10% validation, 10% test
    _, _, test_set = split_dataset(HSERGB, ptrain=0.8, pval=0.1)

    # Create DataLoaders
    test_dl = DataLoader(test_set, shuffle=True, pin_memory=True, drop_last=True)

    return test_dl

# Function to evaluate single sample inference time on test dataset
def test_inference(model, test_loader, device):
    inference_history = []

    start_time = None
    end_time = None
    with torch.no_grad():
        for curr_frame, curr_event_voxels, _ in test_loader:
            # Transfer test data over to GPU
            curr_frame = curr_frame.to(device, non_blocking=True)
            curr_event_voxels = curr_event_voxels.to(device, non_blocking=True)
            
            if device.type == "cuda":
                torch.cuda.synchronize()

            # Measure inference time
            start_time = time.time()
            
            with torch.amp.autocast(device_type=device.type):
                _ = model(curr_frame, curr_event_voxels)
            
            if device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.time()
            
            inference_history.append(end_time - start_time)

    # Print inference time statistics
    print("Teacher model inference statistics:")
    print(f"Min inference time: {min(inference_history)}")
    print(f"Max inference time: {max(inference_history)}")
    print(f"Average inference time: {np.mean(inference_history)}")
    print(f"Inference time standard deviation: {np.std(inference_history)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HS_ERGB Dataset visualization and model test.")
    parser.add_argument("--hs_ergb", type=str, default="../frame_gen/datasets/hs-ergb-dataset")
    parser.add_argument("--teacher_model_path", type=str, default="../frame_gen/common/models/teacher/teacher.pth")
    args = parser.parse_args()

    # Load test dataset (HS-ERGB)
    test_dl = load_test_dataset(pathlib.Path(args.hs_ergb))
    
    # Setup model
    model, device = model_setup(pathlib.Path(args.teacher_model_path))

    # Test teacher model on entire test set
    test_inference(model=model, test_loader=test_dl, device=device)

