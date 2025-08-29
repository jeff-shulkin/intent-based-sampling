'''
Automatically generate synthetic RGB + event camera dataset from given video.
'''

import sys
import os

from typing import Dict

class SyntheticDataset:
    files = None # All listed files in video folder
    
    def __init__(self, video_folder: os.PathLike, settings: Dict[str, int]):
        pass
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx: int):
        pass