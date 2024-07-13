import numpy as np
from utils import *
import pickle

import config
import faiss
import falconn
import scann 


ALGORTIHMS = [
    "FAISS_HNSW",
    "SCANN",
    "FAISS_GT",
    "FALCONN"
]

class BaseANN(object):
    def __init__(self) -> None:
        self.range_algo = False
        self.__params = ""


    def fit(self, dataset):
        """
        Build the index for the data points given in dataset name.
        Assumes that after fitting index is loaded in memory.
        """
        print("This is default 'fit()' function")

    def load_index(self, dataset):
        """
        Load the index for dataset. Returns False if index
        is not available, True otherwise.

        Checking the index usually involves the dataset name
        and the index build paramters passed during construction.
        """
        print("This is default 'load_index()' function")


    def query(self, Q, k):
        """Carry out a batch query for k-NN of on set X."""
        raise NotImplementedError()
    
    def query(self, Q, k):
        """Carry out a single query for k-NN of on set X."""
        raise NotImplementedError()
    
    def range_query(self, Q, rho) -> list:
        """Carry out a batch query for range serach of on set X."""
        raise NotImplementedError()
    
    def range_query(self, Q, rho) -> list:
        """Carry out a single query for range search of on set X."""
        raise NotImplementedError()

    def params(self) -> str:
        return self.__params

    def __str__(self):
        return self.name


class FAISS_HNSW(BaseANN):
    def __init__(self, ef_factor=2, ef_constr = 32) -> None:
        super().__init__()
        self.range_algo = False
        self.M = 32
        self.efSearch_factor = ef_factor
        self.efConstruction = ef_constr

    def fit(self, X, save_index=True):
        self.index = faiss.index_factory(X.shape[1], 'HNSW32', faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = self.efConstruction

    def load_index(self, X):
        self.index.add(X)
        return

    def query_seq(self, Q, k=config.NUM_NEIGHBOURS):
        if k==0:
            return []
        self.index.hnsw.efSearch = int(k*self.efSearch_factor)
        D, I = self.index.search(Q[None, :], int(k))
        return I[0] 

    
    def add(self, X):
        self.index.add(X)
        return 

class SCANN(BaseANN):
    def __init__(self, num_leaves_factor=2, num_leaves_to_search_factor=4, reorder_factor=16) -> None:
        super().__init__()
        self.range_algo = False
        self.nlf = num_leaves_factor
        self.nltsf = num_leaves_to_search_factor
        self.rf = reorder_factor

    def fit(self, X, save_index=True):
        self.nl = self.nlf*int(np.sqrt(X.shape[0]))
        self.X = X
        self.searcher = scann.scann_ops_pybind.builder(X, config.NUM_NEIGHBOURS, "dot_product").tree(
            num_leaves=self.nl, num_leaves_to_search=int(self.nl/self.nltsf), training_sample_size=X.shape[0]).score_ah(
            2, anisotropic_quantization_threshold=0.2).reorder(10*config.NUM_NEIGHBOURS).build()


    def query_seq(self, Q, k=config.NUM_NEIGHBOURS):
        if k==0:
            return []
        I, D = self.searcher.search(Q, final_num_neighbors=k, pre_reorder_num_neighbors=self.rf*k+1)
        return I

    def add(self, X):
        self.X = np.concatenate([self.X, X], axis=0)
        self.searcher = scann.scann_ops_pybind.builder(self.X, config.NUM_NEIGHBOURS, "dot_product").tree(
                    num_leaves=self.nl, num_leaves_to_search=int(self.nl/self.nltsf), training_sample_size=self.X.shape[0]).score_ah(
                    2, anisotropic_quantization_threshold=0.2).reorder(10*config.NUM_NEIGHBOURS).build()
        return 

class FAISS_GT(BaseANN):
    def __init__(self) -> None:
        super().__init__()
        self.range_algo = True

    def fit(self, X, save_index=True):
        print("Not saving index for FAISS")
        pass

    def load_index(self, X):
        self.index = faiss.IndexFlatIP(X.shape[1])
        self.index.add(X)

    def query(self, Q, k=1) -> list:
        D, I = self.index.search(Q, k)
        return I

    def query_seq(self, Q, k=1):
        D, I = self.index.search(Q[None, :], k)
        return I[0]
    
    def range_query_seq(self, Q, rho) -> list:
        lims, D, I = self.index.range_search(Q[None, :], rho)
        return I

    def add(self, X):
        self.index.add(X)
        return 


class FAISS_IVFlat(BaseANN):
    def __init__(self, nlist=32, nprobe=16) -> None:
        super().__init__()
        self.range_algo = True
        self.nlist = nlist
        self.nprobe = nprobe
        self.__params = f"l{nlist}_p{nprobe}"

    def fit(self, X, save_index=True):
        print("Not saving index for FAISS")

    def load_index(self, X):
        self.d = X.shape[1]
        self.quantizer = faiss.IndexFlatIP(self.d)
        self.index = faiss.IndexIVFFlat(self.quantizer, self.d, self.nlist, faiss.METRIC_INNER_PRODUCT)
        self.index.train(X)
        self.index.add(X)
        self.index.nprobe = self.nprobe

    def query(self, Q, k=1) -> list:
        D, I = self.index.search(Q, k)
        return I

    def query_seq(self, Q, k=1):
        D, I = self.index.search(Q[None,:], k)
        return I[0]
    
    def range_query_seq(self, Q, rho=0.8) -> list:
        lins, D, I = self.index.range_search(Q[None, :], rho)
        return I

    def add(self, X):
        self.index.add(X)
        return 

class FALCONN(BaseANN):
    def __init__(self, num_probes_factor=70, num_tables=250) -> None:
        super().__init__()
        self.range_algo = True
        self.__params = None
        self.num_probes = num_probes_factor * num_tables 
        self.num_tables = num_tables

    def fit(self, X, save_index=True):

        self.__params = falconn.get_default_parameters(X.shape[0], X.shape[1])
        self.__params.lsh_family = falconn.LSHFamily.CrossPolytope
        self.__params.l = self.num_tables
        self.__params.num_rotations = 1
        self.__params.seed = 0
        self.__params.num_setup_threads = 0
        self.__params.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
        self.__params.distance_function = falconn.DistanceFunction.EuclideanSquared
        falconn.compute_number_of_hash_functions(17, self.__params)
        self.t = falconn.LSHIndex(self.__params)

    def load_index(self, X):
        self.t.setup(X)
        self.index = self.t.construct_query_object()
        self.index.set_num_probes(self.num_probes)
    
    def range_query_seq(self, Q, rho=0.8) -> list:
        I = self.index.find_near_neighbors(Q, 2 - 2*rho)
        return I
    
    def params(self):
        return f"nt{self.num_tables}_npf{self.num_probes//self.num_tables}"