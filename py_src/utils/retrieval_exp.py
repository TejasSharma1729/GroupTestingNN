import faiss
import torch
from torch import nn
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights
from torchvision.utils import save_image
from torchvision.io import read_image
from torchvision.models import VGG16_Weights
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet50_Weights


import numpy as np
import glob
import os
from PIL import Image as PIL_Image
import argparse

import random
import pickle
import time

random.seed(73)
torch.manual_seed(73)
np.random.seed(73)

import matplotlib.pyplot as plt


SOFTMAX_FEAT = "softmax"
device = None

class CustomData(Dataset):
    """
    CustomData dataset
    """

    def __init__(self, name, dirpath, transform=None):
        super(Dataset, self).__init__()
        self.name = name
        self.dirpath = dirpath
        self.image_paths = CustomData.get_all_files_in_directory(dirpath)
        self.classes = CustomData.get_all_classes_in_directory(dirpath)
        self.classes.sort()
        self.idx_to_class = self.classes.copy()
        self.classes = {k: i for i, k in enumerate(self.classes)}
        self.create_class_dict()
        self.transform = transform
        self.class_dict

        
    @staticmethod
    def get_all_files_in_directory(directory):
        file_paths = []
        
        # Walk through the directory and its subdirectories
        for foldername, subfolders, filenames in os.walk(directory):
            for filename in filenames:
                # Get the full path of each file
                file_path = os.path.join(foldername, filename)
                file_paths.append(file_path)
        # file_paths.sort()
        return file_paths

    def get_all_classes_in_directory(path):
        return os.listdir(path)
    
    @staticmethod
    def feature_extractor(img_path, model):
        try:
            img = read_image(img_path)
        except Exception as e:
            print(f"Caught an exception while opening {img_path}, continuing to next image")
            return None

        if img.shape[0] == 1:
            print("Grayscale")
            img = img.repeat(3,1,1)
            
        if img.shape[0] != 3 and img.shape[0] != 1:
            return None  
        img = transforms(img)
        img = img.unsqueeze(dim=0)
        with torch.no_grad():
            feats = model(img.to(device))
        print(feats.shape)
        return feats

    def __getitem__(self, index):
        # Training images
        try:
            img = read_image(self.image_paths[index])
        except Exception as e:
            print(f"Caught an exception while opening {self.image_paths[index]}, continuing to next image")
            return None
        if img.shape[0] == 1:
            img = img.repeat(3,1,1)
        if img.shape[0] != 3 and img.shape[0] != 1:
            return None
        # print(os.path.basename(os.path.dirname(self.image_paths[index])))
        c = self.classes[os.path.basename(os.path.dirname(self.image_paths[index]))]
        return self.transform(img), c
        

    def __len__(self):
        return len(self.image_paths)


    def create_class_dict(self):
        self.class_dict = {c : [] for c in self.classes}
        print("Classes : ", self.classes)
        for path in self.image_paths:
            c = os.path.basename(os.path.dirname(path))
            try : 
                self.class_dict[c].append(path)
            except Exception as e : 
                print(e)
                print("Encountered path : " , path)
            
    def generate_queries(self, model, images_per_class=10):
        q_classes = []
        feat_list = []
        img_count = 0
        for k, v in self.class_dict.items():
            per_class_img_count = 0 
            for img_p in v:
                if per_class_img_count >= images_per_class:
                    break
                print(img_count)
                f = CustomData.feature_extractor(img_p, model)
                feat_list.append(f)
                q_classes.append(self.classes[k])
                img_count += 1
                per_class_img_count += 1
        
        feat_mat = torch.nn.functional.normalize(torch.cat(feat_list, dim=0))

        return feat_mat, q_classes
                
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieval test')
    parser.add_argument('--dataset-name', type=str, default='imagenet')
    parser.add_argument('--fetchpath', type=str, default="datasets/imagenet/images")
    parser.add_argument('--savepath', type=str, default="datasets/imagenet/output")
    parser.add_argument('--respath', type=str, default="results")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--no-softmax', action="store_true")
    parser.add_argument('--resnet', action="store_true")

    
    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.device%torch.cuda.device_count())) 
    print("Using device", device)
    savepath = os.path.join(args.savepath, "retrieval")
    
    if args.resnet:
        model_raw = torchvision.models.resnet50(pretrained=True).eval().to(device)
        if args.no_softmax:
            model_raw.fc = torch.nn.Identity()
            SOFTMAX_FEAT = "no_" + SOFTMAX_FEAT
            args.dataset_name = "no_sm_" + args.dataset_name
            model = model_raw
        else:
            model = lambda x : torch.nn.functional.softmax(model_raw(x), dim=1)
        transforms = ResNet50_Weights.IMAGENET1K_V1.transforms()
    else:  
        model_raw = torchvision.models.vgg16(pretrained=True).eval().to(device)
        if args.no_softmax:
            model_raw.classifier[6] = torch.nn.Identity()
            SOFTMAX_FEAT = "no_" + SOFTMAX_FEAT
            args.dataset_name = "no_sm_" + args.dataset_name
            model = lambda x : model_raw(x)
        else:
            model = lambda x : torch.nn.functional.softmax(model_raw(x), dim=1)
        transforms = VGG16_Weights.IMAGENET1K_V1.transforms()
        

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)
    
    dataset = CustomData(name="ImageNet", dirpath=args.fetchpath, transform=transforms)
    data_loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=32)
    savepath = os.path.join(savepath, args.dataset_name)
    os.makedirs(savepath, exist_ok = True)
    
    feature_list = []
    class_list= [] 
    save_num = 0 
    
    for idx, (imgs, classes) in enumerate(data_loader):
        with torch.no_grad():
            features = model(imgs.to(device))
        feature_list.append(features)
        class_list.extend(classes)
        if (idx + 1) % 1000 == 0:
            feat_mat = torch.nn.functional.normalize(torch.cat(feature_list, dim=0))
            with open(os.path.join(savepath,  f"{SOFTMAX_FEAT}_{save_num}.npy"), 'wb') as f:
                np.save(f, feat_mat.cpu().numpy())
            save_num += 1
            feature_list.clear()
        
    feat_mat = torch.nn.functional.normalize(torch.cat(feature_list, dim=0))
    with open(os.path.join(savepath, f"{SOFTMAX_FEAT}_{save_num}.npy"), 'wb') as f:
        np.save(f, feat_mat.cpu().numpy())
    save_num += 1
    feature_list.clear()
    
    feature_list = []
    mat_names = list(glob.glob(os.path.join(savepath, f"{SOFTMAX_FEAT}_*.npy")))
    mat_names.sort(key = lambda x : int(x.split(".")[0].split("_")[-1]))
    for idx, r in enumerate(mat_names):
        print(r)
        with open(r, "rb") as f:
            X = np.load(f)
            feature_list.append(X)
            print("Appended")
    feat = np.concatenate(feature_list, axis=0)
    
    print("Total feature shape :", feat.shape)
    
    np.save(os.path.join(savepath, f"X_{SOFTMAX_FEAT}.npy"), feat)
    with open(os.path.join(savepath, f"x_{SOFTMAX_FEAT}_class_labels.pkl"), "wb") as fp:   #Pickling
       pickle.dump(class_list, fp)

    queries, q_classes = dataset.generate_queries(model=model)
    queries = queries.cpu().numpy()
    np.save(os.path.join(savepath, f"Q_{SOFTMAX_FEAT}.npy"), queries)
    
    with open(os.path.join(savepath, f"q_{SOFTMAX_FEAT}_class_labels.pkl"), "wb") as fp:   #Pickling
       pickle.dump(q_classes, fp)
    
    index = faiss.IndexFlatIP(feat.shape[1])
    index.add(feat)
    precision = {}
    recall = {}
    avg_retrieved = {}
    for rho in [0.6,0.7,0.8,0.9]:
        lims, D, I = index.range_search(queries, rho)
        precision[f"rho{rho}"] = 0
        recall[f"rho{rho}"] = 0
        
        for i in range(len(lims) - 1):
            pred = I[lims[i] : lims[i+1]]
            gt = q_classes[i]
            pred_classes = [class_list[p] for p in pred]
            
            total_gt_images = len(dataset.class_dict[dataset.idx_to_class[gt]])
            true_pos = len([i for i in pred if class_list[i]==gt])
            precision[f"rho{rho}"] += true_pos/len(pred_classes) if len(pred_classes) != 0 else 1.0
            recall[f"rho{rho}"] += true_pos/total_gt_images if total_gt_images != 0 else 1.0
        
        precision[f"rho{rho}"] /= len(q_classes)
        recall[f"rho{rho}"] /= len(q_classes)
        avg_retrieved[f"rho{rho}"] = len(I)/len(q_classes)
        print(f"For rho =", rho)
        print(f"Average_precision : {precision[f'rho{rho}']}")
        print(f"Average_recall : {recall[f'rho{rho}']}")
        print(f"Average_retrieved : {avg_retrieved[f'rho{rho}']}")
        
    os.makedirs(os.path.join(args.respath, "retrieval", args.dataset_name), exist_ok=True)
    with open(os.path.join(args.respath, "retrieval", args.dataset_name, "agg.txt"), "w") as f:
        f.write(f"Average_precision : {precision}\n\n")
        f.write(f"Average_recall : {recall}\n\n")
        f.write(f"Average_retrieved : {avg_retrieved}")
        
