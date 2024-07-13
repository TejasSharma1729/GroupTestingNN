import os
import glob
import argparse
from functools import partial

import numpy as np

from sklearn.preprocessing import normalize

from PIL import Image as PIL_Image

import torch
from torch import nn
import torchvision
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ResNet50_Weights
from torchvision.transforms import v2
from torchvision.utils import save_image
from torchvision.io import read_image
from torchvision.models import VGG16_Weights

# Constants
NUM_SAVE_IMAGES = 10
num_gpus = torch.cuda.device_count()

# Determine device to use (GPU or CPU)
if num_gpus == 0:
    device = torch.device("cpu")
else:
    max_free_memory = None
    free_gpu_idx = 0
    for gpu_idx in range(num_gpus):        
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free memory inside reserved
        if max_free_memory is None or max_free_memory < f:
            max_free_memory = f
            free_gpu_idx = gpu_idx
    device = torch.device(f"cuda:{free_gpu_idx}")

NAME = "vgg_feature_mat_"

# Function to get all file paths in a directory and its subdirectories
def get_all_files_in_directory(directory):
    file_paths = []
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            file_paths.append(file_path)
    return file_paths

# Function to print messages based on quiet mode
def print_tmp(quite, txt):
    if quite:
        return
    else:
        return print(txt)

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Script to generate VGG16 embeddings to generate database')
    parser.add_argument('--root', type=str, default='dummy')
    parser.add_argument('--save-path', type=str, default='dummy')
    parser.add_argument('--max-images', type=int, default=-1)
    parser.add_argument('--quite', action="store_true")

    args = parser.parse_args()
    print_mod = partial(print_tmp, args.quite)
    
    # Get all image file paths
    images = get_all_files_in_directory(args.root)
    max_images = args.max_images 
    save_path = args.save_path
    
    # Load VGG16 model
    model_raw = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).eval().to(device)
    model = lambda x: torch.nn.functional.softmax(model_raw(x), dim=1)
    vgg_transforms = VGG16_Weights.IMAGENET1K_V1.transforms()
    
    feat_list = []
    print_mod("Number of images:", len(images))
    
    save_freq = 25000
    file_num = 0
    for idx, img_path in enumerate(images):
        print_mod("Image idx", idx)

        if max_images >= 0 and idx >= max_images:
            break
        try:
            img = read_image(img_path)
        except Exception as e:
            print(f"Caught an exception while opening {img_path}, continuing to next image")
            continue
        
        # Ensure image has 3 channels
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        
        if img.shape[0] != 3:
            continue    
        
        img = vgg_transforms(img)
        img = img.unsqueeze(dim=0)
        with torch.no_grad():
            feats = model(img.to(device))
            
        print_mod(feats.shape) 
            
        feat_list.append(feats)
        
        # Save features periodically
        if (idx + 1) % save_freq == 0:
            feat_mat = torch.cat(feat_list, dim=0)    
            print_mod(feat_mat.shape)
            with open(os.path.join(save_path, f"{NAME}_{file_num}.npy"), 'wb') as f:
                np.save(f, feat_mat)
            file_num += 1
            feat_list.clear()
            
    # Final save of remaining features
    feat_mat = torch.cat(feat_list, dim=0)
    print_mod(feat_mat.shape)
    with open(os.path.join(save_path, f"{NAME}_{file_num}.npy"), 'wb') as f:
        np.save(f, feat_mat)
        
    # Merge all saved feature files
    feat_list = []
    for idx, r in enumerate(glob.glob(os.path.join(save_path, f"{NAME}*.npy"))):
        print_mod(r)
        with open(r, "rb") as f:
            X = np.load(f)
            feat_list.append(X)
            print_mod("Appended")
    feat = np.concatenate(feat_list, axis=0)
    
    print_mod("Total feature shape:", feat.shape)
    
    np.savetxt(os.path.join(args.save_path, "X.txt"), normalize(feat, axis=1, norm='l2'), delimiter=",")
