#!/usr/bin/python3

import sys # System handling

# First detect whether a virtual environment is active
def is_venv_active():
    return sys.prefix != sys.base_prefix

if is_venv_active():
    # Test whether all packages are installed
    try:
        import torch
        import torchvision
        import torchdistill
        import timm
        import socket
        import time
        import numpy
        import scipy
        import sklearn
        import skimage
        import imutils
        import lpips
        import cv2


    except ImportError as e:
        print(f"Test failed on package {e.name}. Make sure to install it in your virtual environment!")
        sys.exit(1)

    print("All relevant Python packages successfully found.")

    # Test whether GPU is being used or not.
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device {device}....")

else:
    print("No virtual environment active.")
