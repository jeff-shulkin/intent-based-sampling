import pathlib
import argparse
import numpy as np
import cv2
from torch.utils.data import DataLoader
from frame_gen.datasets.HS_ERGB_dataset import HSERGBDataset

def visualize_HSERGB(HSERGB_path: pathlib.Path):

    # Load dataset
    print("Loading HS-ERGB dataset...")
    hs_ergb = HSERGBDataset(HSERGB_path)
    hs_ergb_loader = DataLoader(hs_ergb, batch_size=1, shuffle=False)
    print("HS-ERGB dataset loaded.")

    print(f"HS-ERGB sample count: {len(hs_ergb)}")

    window_name = "HS-ERGB Video Frame"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def tensor_to_rgb(rgb_tensor):
        img_tensor = rgb_tensor.squeeze(0) # remove batch dimension
        img_np = img_tensor.permute(1, 2, 0).numpy()
        return (img_np * 255).astype("uint8")

    # Visualize the RGB video-stream to confirm that it was loaded properly
    for idx, (frame_t, events, frame_tp1) in enumerate(hs_ergb_loader):
        img = tensor_to_rgb(frame_t)

        cv2.imshow(window_name, img)
    
        # wait a bit to simulate video
        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HS_ERGB Dataset visualization and model test.")
    parser.add_argument("--hs_ergb", type=str, default="../frame_gen/datasets/hs-ergb-dataset")
    args = parser.parse_args()
    visualize_HSERGB(pathlib.Path(args.hs_ergb).expanduser().resolve())