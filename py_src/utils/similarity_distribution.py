import os
import random
import argparse 

import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.preprocessing import normalize

import scipy.stats as ss


random.seed(73)

# constants 
BINS=10

parser = argparse.ArgumentParser(description='Creating min and max indices for pooling based group testing')
parser.add_argument('--dataset-name', type=str, default='dummy')
parser.add_argument('--fetchpath', type=str, default='./output')
parser.add_argument('--savepath', type=str, default='./output')


name = "VGG"

args = parser.parse_args()
dataset_name = args.dataset_name + f"_{name}_"
fetchpath = args.fetchpath
if args.savepath is None:
    args.savepath = args.fetchpath
os.makedirs(args.savepath, exist_ok=True)
    
X = np.loadtxt(open(os.path.join(fetchpath, "X.txt"), "rb"), delimiter=",")
X_small = X[random.sample(range(X.shape[0]), 1000),:].copy() #Randomly sample 1000 points from the dataset

print("X shape", X.shape)
print("Q shape", X_small.shape)

sns.set(style="darkgrid")

## Dot products
dot_prod = np.matmul(normalize(X, axis=1, norm='l2'), normalize(Q, axis=1, norm='l2').transpose()).flatten()

params = ss.expon.fit(dot_prod)
print("Similarity distribution params :",params)
rX = np.linspace(0,1, BINS)
rP = ss.expon.pdf(rX, *params)


plt.plot(rX, rP)
sns.histplot(data=pd.DataFrame({"Similarity" : dot_prod}), x="Similarity",  bins = BINS, kde=True, stat="probability")
plt.title(f"Similarity distribution {name}")
plt.savefig(os.path.join(args.savepath, dataset_name + "similarity_hist.png"))

