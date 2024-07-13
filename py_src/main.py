
# Copyright (c) Harsh-Sensei

import argparse
import collections
import json
import os
import random
import sys
import time
import numpy as np
import pickle

import datasets
import algorithms
from utils import *
import config

import tqdm

# DEF_DATASETS = ["ImageNet", "InstaCities", "IMDBWiki", "GLAMI", "MIRFLICKR"]
DEF_DATASETS = datasets.DATASETS
DEF_ALGORITHMS = algorithms.ALGORTIHMS

QUITE_MODE = os.getenv("QUITE_MODE", default=False)

# Global variable to store the ground truth result from FAISS
GT_RES = None


## Setting parameters to try by each search algorithm
FAISS_IVF_PARAMS_NP = [2,16,32]
FAISS_IVF_PARAMS_NL = [32, 64, 128]

FALCONN_PARAMS_NT = [100, 150, 200, 250, 300]
FALCONN_PARAMS_NPF = [40, 50, 70, 100]

SCANN_PARAMS_NLF = [2]
SCANN_PARAMS_NLTSF = [2, 4]
SCANN_PARAMS_RF = [4, 8, 16]

HNSW_PARAMS_EFS = [2, 8, 32]
HNSW_PARAMS_EFC = [32, 64]


def argument_parser():
    parser = argparse.ArgumentParser(description='Approximate Nearest Neighbour Search')
    parser.add_argument('--data-dir', type=str, default="datasets")
    parser.add_argument('--results-dir', type=str, default="results")
    parser.add_argument('--datasets', nargs='+', type=str, default=DEF_DATASETS)
    parser.add_argument('--algorithms', nargs='+', type=str, default=DEF_ALGORITHMS)
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything')
    parser.add_argument('--try-params', action='store_true')
    parser.add_argument('--streaming', action='store_true')
    parser.add_argument('--runbook', type=str, default="./streaming_runbook.json")
    return parser


def run(algo, dataset, dataset_n="dummy", gt_path="./ground_truth"):
    global GT_RES
    check_dir(gt_path)
    
    if algo.__class__.__name__ == "FAISS_GT":
        if os.path.exists(os.path.join(gt_path, dataset_n + ".gt.pkl")):
            with open(os.path.join(gt_path, dataset_n + ".gt.pkl"), 'rb') as f:
                GT_RES = pickle.load(f)
            return 

    ret_dict = {"MQT" : None, f"AvgRecall" : None, f"AvgPrecision" : None}
        
    algo.fit(dataset.X)
    algo.load_index(dataset.X)

    start_time = time.time()
    if algo.range_algo:
        res = []
        for elem in dataset.Q:
            res.append(algo.range_query_seq(elem, rho=config.RHO))

    elif algo.knn_algo:
        res = []
        for idx, elem in enumerate(dataset.Q):
            res.append(algo.query_seq(elem, k=dataset.K[idx]))
    else:
        raise Exception(f"{algo.__class__.__name__} is not range based or knn based")

    end_time = time.time()
    if algo.__class__.__name__ == "FAISS_GT":
        GT_RES = res
        with open(os.path.join(gt_path, dataset_n + ".gt.pkl"), "wb") as f:
            pickle.dump(GT_RES, f)
    ret_dict["QT"] = end_time - start_time
    ret_dict["MQT"] = (end_time - start_time) / dataset.Q.shape[0]
    ret_dict[f"AvgRecall"] = avg_recall(GT_RES, res)
    ret_dict[f"AvgPrecision"] = avg_precision(GT_RES, res)

    return ret_dict


def stream_run(algo, dataset, dataset_n="dummy", run_book="./streaming_runbook.json", init_fraction=0.8, gt_path="./ground_truth"):
    global GT_RES
    check_dir(gt_path)
    
    if algo.__class__.__name__ == "FAISS_GT":
        tmp_path = os.path.exists(os.path.join(gt_path, dataset_n + "_streaming_" + str(config.RHO) + ".gt.pkl"))
        if tmp_path:
            with open(tmp_path, 'rb') as f:
                GT_RES = pickle.load(f)
            return
         
    streaming_steps = json.load(open(run_book, "r"))    
    ret_dict = {"IndexTime" : None, "MQT" : None, f"AvgRecall" : None, f"AvgPrecision" : None}
    preds = []
    running_idx = int(init_fraction*dataset.X.shape[0])
    
    index_start_time = time.time()
    algo.fit(dataset.X[:running_idx], save_index=True)
    algo.load_index(dataset.X[:running_idx])
    index_elapsed_time = time.time() - index_start_time

    total_query_time, num_query_actions = 0, 0
    total_index_time, num_index_actions = 0, 0
    
    streaming_start_time = time.time()

    for step in range(len(streaming_steps)):
        action = streaming_steps[str(step)]["action"]
        value = streaming_steps[str(step)]["value"]
        if action == "search":
            num_query_actions += 1
            q_start_time = time.time()
            if algo.range_algo:
                res = algo.range_query_seq(dataset.Q[value], rho=config.RHO)
            elif algo.knn_algo:
                res = algo.query_seq(dataset.Q[value], k=len(GT_RES[num_query_actions-1]))
            else:
                raise Exception(f"{algo.__class__.__name__} is not range based or knn based")
            q_time = time.time() - q_start_time
            preds.append(res)
            total_query_time += q_time
            
        elif action == "insert":
            num_index_actions += 1
            i_start_time = time.time()
            algo.add(dataset.X[running_idx: (running_idx+value)])
            i_time = time.time() - i_start_time
            running_idx += value
            total_index_time += i_time
            
        else :
            raise Exception("Given action in streaming runbook does not exists")
        
    streaming_time = time.time() - streaming_start_time
    

    if algo.__class__.__name__ == "FAISS_GT":
        GT_RES = preds
        with open(os.path.join(gt_path, dataset_n + str(config.RHO) + ".gt.pkl"), "wb") as f:
            pickle.dump(GT_RES, f)
    
    ## store results
    ret_dict[f"AvgRecall"] = avg_recall(GT_RES, preds)
    ret_dict[f"AvgPrecision"] = avg_precision(GT_RES, preds)
    ret_dict["MQT"] = total_query_time/num_query_actions
    ret_dict["IndexTime"] = total_index_time/num_index_actions
    ret_dict["TotalStreamingTime"] = streaming_time
    ret_dict["NumQueryActions"] = num_query_actions
    ret_dict["NumIndexActions"] = num_index_actions
    ret_dict["IndexInitTime"] = index_elapsed_time
    
    return ret_dict



if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()

    ## Initilizations
    seed_everything(args.seed)
    if QUITE_MODE:
        blockPrint()

    m_datasets = {}
    m_algos = {}

    dne_datasets = [] ## Does not exists datasets
    dne_algos = [] ## Does not exists algos

    ## Loading datasets and algorithm classes
    for dataset in args.datasets:
        if dataset in vars(datasets):
            m_datasets[dataset] = vars(datasets)[dataset](args.data_dir)
        else: 
            dne_datasets.append(dataset)
    
    if "FAISS_GT" != args.algorithms[0]:
        if "FAISS_GT" in args.algorithms:
            args.algorithms.remove("FAISS_GT")
        args.algorithms = ["FAISS_GT"] + args.algorithms

    if args.try_params:
        for algo in args.algorithms:
            if algo in vars(algorithms):
                if algo == "FAISS_IVFlat":
                    for nprobe in FAISS_IVF_PARAMS_NP:
                        for nlist in FAISS_IVF_PARAMS_NL:
                            m_algos[algo + f"_nl{nlist}_np{nprobe}"] = vars(algorithms)[algo](nprobe=nprobe, nlist=nlist)
                elif algo == "FALCONN":
                    for nt in FALCONN_PARAMS_NT:
                        for npf in FALCONN_PARAMS_NPF:
                            m_algos[algo + f"_nt{nt}_npf{npf}"] = vars(algorithms)[algo](num_probes_factor=npf, num_tables=nt)
                elif algo == "SCANN":
                    for nlf in SCANN_PARAMS_NLF:
                        for nltsf in SCANN_PARAMS_NLTSF:
                            for rf in SCANN_PARAMS_RF: 
                                m_algos[algo + f"_nlf{nlf}_nltsf{nltsf}_rf{rf}"] = vars(algorithms)[algo](num_leaves_factor=nlf, num_leaves_to_search_factor=nltsf, reorder_factor=rf)
                elif algo == "FAISS_HNSW":
                    for efs in HNSW_PARAMS_EFS:
                        for efc in HNSW_PARAMS_EFC:
                            m_algos[algo + f"_efs{efs}_efc{efc}"] = vars(algorithms)[algo](ef_factor=efs, ef_constr=efc)
                else:
                        m_algos[algo] = vars(algorithms)[algo]()
            else: 
                dne_algos.append(algo)
        
    else:
        for algo in args.algorithms:
            if algo in vars(algorithms):
                m_algos[algo] = vars(algorithms)[algo]()
            else: 
                dne_algos.append(algo)

    ## Reporting the datasets and algos that are not implemented
    if len(dne_datasets) != 0:
        t_str = ",".join(dne_datasets)
        print(f"Datasets : {t_str} :: Are not implemented")

    if len(dne_algos) != 0:
        t_str = ",".join(dne_algos)
        print(f"Algorithms : {t_str} :: Are not implemented")
    
    ## Setting up datasets
    for dataset_name, dataset in m_datasets.items():
        dataset.download_and_extract_dataset()
        dataset.loadDatasetFromCSV()

    all_res = {} # all results
    
    ## Running the algorithms on required datasets
    for dataset_name, dataset in m_datasets.items():
        for algo_name, algo in m_algos.items():
            
            check_dir(os.path.join(args.results_dir, algo_name))
            print(f"Running algorithm : {algo_name} with dataset : {dataset_name}")
            if args.streaming:
                result_dict = stream_run(algo, dataset, dataset_n=dataset_name, run_book=args.runbook)
            else:    
                result_dict = run(algo, dataset, dataset_n=dataset_name)
                
            print(result_dict)
            print()
            k_num_list = [len(elem) for elem in GT_RES]
            try : 
                print(f"Avg K for {dataset_name} :", sum(k_num_list)/len(k_num_list))
            except Exception as e:
                print(e)
                
            with open(os.path.join(args.results_dir, algo_name, dataset_name) + f"_rho{config.RHO}.txt", 'w') as f:
                dump_str = json.dumps(result_dict, sort_keys=True)
                f.write(dump_str + "\n")
                all_res[f"{algo_name}_{dataset_name}"] = dump_str
            
    ## Dumping all resuts in results.txt
    dump_str = json.dumps(all_res, sort_keys=True)
    with open(os.path.join(args.results_dir, "results.txt"), 'w') as f:
        f.write(dump_str + "\n")
    
    print("Script by Harsh-Sensei")
            