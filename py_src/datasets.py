import numpy as np
import os
import h5py

DATASETS = [
    "MIRFLICKR",
    "ImageNet",
    "IMDBWiki",
    "InstaCities",
    "DummyData",
]


def check_dir(path, create_dir=True):

    if os.path.exists(path):
        return True
    
    if create_dir:
        try: 
            os.mkdir(path) 
        except OSError as error: 
            print(error)  

    return False


class BaseDataset:
    def __init__(self, data_dir):
        self.num_datapoints = 10000
        self.dim = 256
        self.dataset_url = "dummy"
        self.location = os.path.join(data_dir, "dummy")
        self.X = None
        self.Q = None

    def getNumItems(self):
        return self.num_datapoints
    
    def getDim(self):
        return self.dim
    
    def download_and_extract_dataset(self):
        raise NotImplementedError

    def loadDataset(self):
        raise NotImplementedError

    def numQueries(self):
        return np.shape(self.Q)[0]
    
    def loadDatasetFromCSV(self):
        with open(os.path.join(self.location, f'X.txt'), 'rb') as f:
            self.X = np.loadtxt(f, delimiter=",") 
        with open(os.path.join(self.location, f'Q.txt'), 'rb') as f:
            self.Q = np.loadtxt(f, delimiter=",")
        self.num_datapoints = self.X.shape[0]
        self.dim = self.X.shape[1]
        
        print("Dataset loaded")
        print("X :", self.X.shape)
        print("Q :", self.Q.shape)
        
        return



class MIRFLICKR(BaseDataset):
    # url = https://press.liacs.nl/mirflickr/mirdownload.html 
    def __init__(self, data_dir):
        self.d_name = "mirflickr" # directory name 
        self.location = os.path.join(data_dir, self.d_name, "output") 
        

class ImageNet(BaseDataset):
    # url = https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data 
    def __init__(self, data_dir):
        self.d_name = "imagenet"
        self.location = os.path.join(data_dir, self.d_name, "output")
    

class IMDBWiki(BaseDataset):
    # url = https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
    def __init__(self, data_dir):
        self.d_name = "imdb_wiki"
        self.location = os.path.join(data_dir, self.d_name, "output")


class InstaCities(BaseDataset):
    # url = https://gombru.github.io/2018/08/01/InstaCities1M/ 
    def __init__(self, data_dir):
        self.d_name = "insta_1m"
        self.location = os.path.join(data_dir, self.d_name, "output")


class DummyData(BaseDataset):
    def __init__(self, data_dir):
        self.num_datapoints = 10000
        self.dim = 256
        self.dataset_url = "dummy"
        self.d_name = "dummy"
        self.location = os.path.join(data_dir, "dummy")
        self.X = None
    
    def download_and_extract_dataset(self):
        exists = check_dir(self.location)
        if not exists:
            print("Download the dataset :", self.__class__.__name__)
            mat = np.random.rand(self.num_datapoints, self.dim)
            mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
            queries = np.random.rand(int(0.01*self.num_datapoints),self.dim )
            queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

            with open(os.path.join(self.location, 'dataset.npy'), 'wb') as f:
                np.save(f, mat)
            with open(os.path.join(self.location, 'queries.npy'), 'wb') as f:
                np.save(f, queries)

    def loadDataset(self, run_knn=[False, "./"]):
        with open(os.path.join(self.location, 'dataset.npy'), 'rb') as f:
            self.X = np.load(f)   
        with open(os.path.join(self.location, 'queries.npy'), 'rb') as f:
            self.Q = np.load(f)
        if run_knn[0]:
            self.K = np.genfromtxt(os.path.join(run_knn[1], self.d_name, "kthresholds.txt"), delimiter=',').astype(np.int32)


