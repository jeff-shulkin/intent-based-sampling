import torch
import torch.nn as nn
import torch.nn.functional as F

class NextFrameTransformer(nn.Transformer):
    """
    This model is the "Teacher" transformer model, whose goal is to predict any N number of frames based on 
    """