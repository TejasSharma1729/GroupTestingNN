import json
import random 
import argparse

random.seed(73)
N = 520_000
MAX_SUM = int(0.2*N)
INSERTION_RANGE = 200

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creating runbook for streaming experiments')
    parser.add_argument('--outfile', type=str, default='streaming_runbook.json')
    args = parser.parse_args()
    
    runbook_dict = {}
    curr_sum = 0
    runbook_idx = 0
    while curr_sum < MAX_SUM:
        if random.randint(0,2) == 0:
            temp_dict = {}
            temp_dict["action"] = "insert"
            temp_len = random.randint(INSERTION_RANGE//2,INSERTION_RANGE)
            curr_sum += temp_len
            temp_dict["value"] = temp_len
            runbook_dict[runbook_idx] = temp_dict
        else :
            temp_dict = {}
            temp_dict["action"] = "search"
            temp_dict["value"] = random.randint(0,9999)
            runbook_dict[runbook_idx] = temp_dict
        
        runbook_idx += 1
    
    with open(args.outfile, 'w') as fp:
        json.dump(runbook_dict, fp)
            