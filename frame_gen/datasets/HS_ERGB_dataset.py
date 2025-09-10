import torch 
from torchvision import transforms 
import numpy as np 
from torch.utils.data import Dataset 
from PIL import Image 
import pathlib 
import os
from frame_gen.common.models.modules import Event_ToTensor

class HSERGBDataset(Dataset): 
    """ Pytorch class for HS-ERGB Dataset. """ 
    def __init__(self, root: pathlib.Path, image_size=(224, 224)): 
        # Expand to use solely absolute path
        root = root.expanduser().resolve()
         
        self.samples = []
        self.image_size = image_size

        # Define RGB and event transforms
        self.rgb_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

        self.event_transform = transforms.Compose([
            Event_ToTensor()
        ])

        # find all scenes 
        scenes = []
        for category in ["close", "far"]:
            category_path = root / "hsergb" / category / "test"
            if category_path.exists():
                scene_dirs = [d for d in category_path.iterdir() if d.is_dir()]
                scenes.extend(scene_dirs)
        
        for scene in scenes:
            events_dir = scene / "events_aligned"
            images_dir = scene / "images_corrected"
            ts_path = images_dir / "timestamp.txt" 
            if not (events_dir.exists() and images_dir.exists() and ts_path.exists()): 
                continue

            frame_ts = np.loadtxt(ts_path)
            img_files = list(images_dir.glob("*.png"))
            
            # gather event files (00001.npz, 00002.npz, …) 
            event_files = sorted(events_dir.glob("*.npz"), key=lambda p: int(p.stem))

            # keep only consecutive frame pairs (frame[i], frame[i+1]) 
            for i in range(len(frame_ts) - 1): 
                self.samples.append({ 
                    "scene": scene.name, 
                    "frame_t_path": img_files[i], 
                    "frame_tp1_path": img_files[i+1], 
                    "ts_t": frame_ts[i], "ts_tp1": frame_ts[i+1], 
                    "event_files": event_files, }) 
                
    def __len__(self): 
        return len(self.samples) 
    
    def __getitem__(self, idx): 
        s = self.samples[idx] 
        # load frames 
        def load_frame(path): 
            return Image.open(path).convert("RGB") 

        # load events within [ts_t, ts_tp1)
        events_list = []
        for ef in s["event_files"]:
            with np.load(ef, mmap_mode="r") as ev:
                t = ev["t"]
                mask = (t >= s["ts_t"]) & (t < s["ts_tp1"])
                if np.any(mask):
                    x = ev["x"][mask]
                    y = ev["y"][mask]
                    p = ev["p"][mask]
                    events_list.append(np.stack([t[mask], x, y, p], axis=1)) 
                
        if events_list:
            events = np.concatenate(events_list, axis=0)
        else:
            events = np.empty((0,4)) 

        # Load and normalize RGB frames
        frame_t = self.rgb_transform(load_frame(s["frame_t_path"]))
        frame_tp1 = self.rgb_transform(load_frame(s["frame_tp1_path"]))

        # Convert events to normalized voxels
        event_voxels = self.event_transform(events)
            
        # Return frame, events until next frame, and next frame as tensors
        return frame_t, event_voxels, frame_tp1
