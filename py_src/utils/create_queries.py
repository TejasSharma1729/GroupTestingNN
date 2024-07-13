import os
import argparse
import random
from PIL import Image as PIL_Image

import numpy as np

from sklearn.preprocessing import normalize

import torch
from torch import nn
import torchvision
from torchvision.utils import save_image
from torchvision.io import read_image
from torchvision.models import VGG16_Weights

random.seed(73)

num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    device = torch.device("cpu")
else :
    max_free_memory = None
    free_gpu_idx = 0
    for gpu_idx in range(num_gpus):        
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved
        if max_free_memory is None:
            max_free_memory = f
            free_gpu_idx = gpu_idx
        if max_free_memory < f :
            max_free_memory = f
            free_gpu_idx = gpu_idx
    device = torch.device("cuda:"+str(free_gpu_idx)) 


def get_all_files_in_directory(directory):
    file_paths = []
    
    # Walk through the directory and its subdirectories
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            # Get the full path of each file
            file_path = os.path.join(foldername, filename)
            file_paths.append(file_path)
    
    return file_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create augmented images and extract VGG features')
    parser.add_argument('--root', type=str, default='dataset')
    parser.add_argument('--save-path', type=str, default='output') 
    parser.add_argument('--max-images', type=int, default=10_000)
    args = parser.parse_args()
    
    max_images = args.max_images 
    save_path = args.save_path
    
    images = random.sample(get_all_files_in_directory(args.root), 2*max_images)

    model_raw = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).eval().to(device)
    model = lambda x : torch.nn.functional.softmax(model_raw(x), dim=1)
    vgg_transforms = VGG16_Weights.IMAGENET1K_V1.transforms()

    scale = 2
    augmentations = torchvision.transforms.Compose([
        torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(3)], p=0.5),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.5),
        torchvision.transforms.RandomApply([torchvision.transforms.RandomResizedCrop(size=(224//scale, 224//scale))], p=0.2),
        torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation(3)], p=0.2)
    ])

    feat_list = []
    num_imgs = 0
    for idx, img_path in enumerate(images):
        if num_imgs >= max_images:
            break
        try:
            img = read_image(img_path)
            if img.shape[0] == 1:
                print(f"Image {idx} is grayscale : ", img_path)
                img = img.repeat(3,1,1)
            if img.shape[0] == 4:
                print("Dimension 0 is 4")
                print("Continuing to next image")
                continue
            
            aug_img = augmentations(img)
            aug_img = aug_img.unsqueeze(dim=0)
            aug_img = vgg_transforms(aug_img)
            
            with torch.no_grad():
                feats = model(aug_img.to(device))
            feat_list.append(feats)
            
        except Exception as e:
            print(f"Caught an exception {e}; while running {img_path}, continuing to next image")
            continue
        
        num_imgs += 1
        
    feat_mat = torch.cat(feat_list, dim=0)
    
    print("Saving matrix of shape :", feat_mat.shape)
    np.savetxt(os.path.join(save_path, "Q.txt"), normalize(feat_mat.cpu().numpy()), delimiter=",")
    print("Successfully saved the matrix to", os.path.join(save_path, "Q.txt"))
    
    