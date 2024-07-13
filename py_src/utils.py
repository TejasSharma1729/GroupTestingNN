import sys
import os
import random
import numpy as np
import config
import faiss

# Disable printing
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore printing
def enablePrint():
    sys.stdout = sys.__stdout__

# Creating directories
def check_dir(path: str, create_dir=True):

    if os.path.exists(path):
        return True
    
    if create_dir:
        try: 
            os.mkdir(path) 
        except OSError as error: 
            print(error)  

    return False

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def recall(gt, predicted, k):
    mean_recall = 0
    num_items = len(gt)
    for s1, s2 in zip(gt, predicted):
        assert len(s1) == k
        mean_recall += len(set(s1) & set(s2))/k
    
    return mean_recall/num_items

def avg_recall(gt, predicted):
    mean_recall = 0
    num_items = len(gt)
    for idx, (s1, s2) in enumerate(zip(gt, predicted)):
        mean_recall += len(set(s1) & set(s2))/len(s1) if len(s1) != 0 else 1
    
    return mean_recall/num_items

def avg_precision(gt, predicted):
    mean_recall = 0
    num_items = len(gt)
    for idx, (s1, s2) in enumerate(zip(gt, predicted)):
        mean_recall += len(set(s1) & set(s2))/len(s2) if len(s2) != 0 else 1
    
    return mean_recall/num_items

def createQueryThres(dataset):
    if os.path.exists(os.path.join(dataset.location, f"query_thres_{config.NUM_NEIGHBOURS}.npy")):
        return 0
    index = faiss.IndexFlatIP(dataset.dim)
    index.add(dataset.X)

    D, I = index.search(dataset.Q, config.NUM_NEIGHBOURS+1)
    thres = np.array([elem[-1] for elem in D])

    with open(os.path.join(dataset.location, f"query_thres_{config.NUM_NEIGHBOURS}.npy"), 'wb') as f:
        np.save(f, thres)