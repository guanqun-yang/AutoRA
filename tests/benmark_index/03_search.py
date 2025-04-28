import os
import time
import faiss
import psutil
import argparse
import threading
import ir_measures

import numpy as np
import pandas as pd

from tqdm import trange
from ir_measures import *
from utils.common import load_pickle_file

from transformers import set_seed

##################################################

set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--index_type", default="Flat", choices=["Flat", "IVFFlat", "IVFPQ"])
parser.add_argument("--n_list", type=int)
args = parser.parse_args()

TOP_K = 100
BATCH_SIZE = 32
N_LIST = args.n_list
N_PROBE = 200

METRICS = [RR, NDCG@5, NDCG@10, NDCG@20, NDCG@50, NDCG@100, R@5, R@10, R@20, R@50, R@100]

##################################################
peak_memory = 0

def track_memory(stop_event):
    global peak_memory
    process = psutil.Process(os.getpid())
    while not stop_event.is_set():
        mem = process.memory_info().rss / 1024 / 1024  # MB
        if mem > peak_memory:
            peak_memory = mem
        time.sleep(0.1)

##################################################

qrel = load_pickle_file("qrel.pkl")

def reload_and_search(index_type):
    global peak_memory
    peak_memory = 0
    stop_event = threading.Event()
    memory_thread = threading.Thread(target=track_memory, args=(stop_event,))
    memory_thread.start()

    time1 = time.time()
    query_index = faiss.read_index(f"{index_type}_{N_LIST}_query.index")
    doc_index = faiss.read_index(f"{index_type}_{N_LIST}_doc.index")

    doc_index.nprobe = N_PROBE

    total_queries = query_index.ntotal
    all_query_ids = faiss.vector_to_array(query_index.id_map)

    results = []
    for start in trange(0, total_queries, BATCH_SIZE, desc=f"Searching {index_type}..."):
        end = min(start + BATCH_SIZE, total_queries)
        query_vecs = np.array([query_index.index.reconstruct(i) for i in range(start, end)], dtype='float32')
        scores, indices = doc_index.search(query_vecs, TOP_K)

        results += [
            (str(all_query_ids[start + i]), str(indices[i][j]), -scores[i][j])
            for i in range(scores.shape[0])
            for j in range(TOP_K)
            if indices[i][j] != -1
        ]

    time2 = time.time()
    stop_event.set()
    memory_thread.join()

    run = pd.DataFrame(results, columns=["query_id", "doc_id", "score"])
    metrics = ir_measures.calc_aggregate(METRICS, qrel, run)

    return {
        "Index Type": index_type,
        "N_LIST": N_LIST,
        "Search Time": time2 - time1,
        "Peak Memory Reload (MB)": peak_memory,
        **{str(m): metrics[m] for m in METRICS}
    }


result = reload_and_search(args.index_type)
print(result)