# Continue with regular imports
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from commons import get_model, get_tensor

import re
import glob
import os

from PIL import Image 
import cv2 as cv
import shutil
import glob
from pathlib import Path
from PIL import Image

cudnn.benchmark = True
plt.ion()   # interactive mode

train_glob = glob.glob("./images/train/*/*")
val_glob = glob.glob("./images/val/*/*")
test_glob = glob.glob("./images/test/*/*")
total_glob = []
total_glob.append(train_glob)
total_glob.append(test_glob)
total_glob.append(val_glob)

for i, elem in enumerate(train_glob):
    p = Path(elem)
    split = os.path.splitext(p)
    ext = split[1]
    print(ext)
    if ext != "jpg":
        filename_replace_ext = p.with_suffix('')
        # remove last two
        filename_replace_ext = str(filename_replace_ext)[:len(str(filename_replace_ext))-1]
        print(p)
        train_glob[i] = filename_replace_ext

for e in train_glob:
    op = Image.open(e)
    Image.save(op)


data_transforms = {
'train': transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

}

data_dir = './images/'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) for x in ['train', 'val', 'test']}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes
def get_prediction(image_bytes):

    model = get_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image = get_tensor(image_bytes)

    with torch.no_grad():
        outputs = model(image)
        preds = torch.max(outputs, 1)
        print(preds)
        for j in range(1):
            prediction=  preds
    return prediction
    
        
