import numpy as np
import cv2
import time
import torch
from torch.utils.data import DataLoader

from frame_gen.common.models.teacher.model import NextFrameTransformerTeacher
from frame_gen.datasets.HS_ERGB_dataset import HSERGBDataset
from tools.pytorch_tools import determine_device, split_dataset

import pathlib
import argparse

# Function to set up model for inference testing
def model_setup(model_weights_path: pathlib.Path):
    # Resolve model path
    model_weights_path = model_weights_path.expanduser().resolve()

    # Load model
    state_dict = torch.load(model_weights_path, weights_only=False)
    model = NextFrameTransformerTeacher(
        image_size=(224,224),
        patch_size=16,
        embed_dim=512,
        nhid=2048,
        nhead=8,
        nlayers=8,
        dropout=0.1,
        num_voxels=5)
    model.load_state_dict(state_dict)


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

def show_frame_cv2(prev_frame, predicted_frame, window_name="Prev vs Generated"):
    def to_numpy_img(tensor_img):
        # remove batch dim if present
        if tensor_img.dim() == 4 and tensor_img.size(0) == 1:
            tensor_img = tensor_img.squeeze(0)
        img = tensor_img.detach().cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)  # normalize 0-1
        img = (img * 255).astype("uint8")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    prev_img = to_numpy_img(prev_frame)
    pred_img = to_numpy_img(predicted_frame)

    # Make sure sizes match before concatenating
    if prev_img.shape != pred_img.shape:
        h = min(prev_img.shape[0], pred_img.shape[0])
        w = min(prev_img.shape[1], pred_img.shape[1])
        prev_img = cv2.resize(prev_img, (w, h))
        pred_img = cv2.resize(pred_img, (w, h))

    combined = np.hstack((prev_img, pred_img))  # side-by-side
    cv2.imshow(window_name, combined)
    cv2.waitKey(1)

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
            
            predicted_frame = None
            with torch.amp.autocast(device_type=device.type):
                predicted_frame = model(curr_frame, curr_event_voxels)
            
            if device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.time()

            inference_history.append(end_time - start_time)

            # Display generated image:
            show_frame_cv2(predicted_frame, "Generated Frame")
            

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

