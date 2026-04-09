"""Classification components
"""

import os, time, gc, math
import numpy as np
from sklearn.metrics import f1_score as sk_f1
from typing import Tuple, Optional, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import Counter
import wandb

from models.vgg11 import VGG11Encoder, VGG11, init_weights
from models.layers import CustomDropout


class VGG11Classifier(nn.Module):
    def __init__(self, num_classes=37, in_channels=3, dropout_p=0.3, use_batch_norm=True):
        super().__init__()
        self.backbone   = VGG11(in_channels, use_batch_norm)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )
        init_weights(self.classifier)

    def forward(self, x):
        return self.classifier(self.backbone(x))
