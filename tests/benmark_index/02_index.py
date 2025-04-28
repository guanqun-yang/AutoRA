import os
import gc
import time
import psutil
import faiss
import argparse

import threading
import argparse
from utils.common import load_pickle_file

##################################################
# HYPERPARAMETERS
##################################################


parser = argparse.ArgumentParser()
parser.add_argument("--index_type", default="Flat", choices=["Flat", "IVFFlat", "IVFPQ"])
parser.add_argument("--n_list", type=int)
args = parser.parse_args()

TOP_K = 100
BATCH_SIZE = 32
M_PQ = 64
N_LIST = args.n_list
BITS_PER_CODE = 8



start_time = time.time()

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
# INDEX FUNCTIONS
##################################################

query_embeddings = load_pickle_file('query_embeddings.pkl')
doc_embeddings = load_pickle_file('doc_embeddings.pkl')
query_int_ids = load_pickle_file('query_int_ids.pkl')
doc_int_ids = load_pickle_file('doc_int_ids.pkl')

def build_index(embeddings, int_ids, index_type):
    dim = embeddings.shape[1]

    if index_type == "Flat":
        base_index = faiss.IndexFlatL2(dim)
    elif index_type == "IVFFlat":
        quantizer = faiss.IndexFlatL2(dim)
        base_index = faiss.IndexIVFFlat(quantizer, dim, N_LIST)
        base_index.train(embeddings)
    elif index_type == "IVFPQ":
        quantizer = faiss.IndexFlatL2(dim)
        base_index = faiss.IndexIVFPQ(quantizer, dim, N_LIST, M_PQ, BITS_PER_CODE)
        base_index.train(embeddings)
    else:
        raise ValueError("Unknown index type")

    index = faiss.IndexIDMap(base_index)
    index.add_with_ids(embeddings, int_ids)
    return index


def build_and_save_indexes(index_type):
    global peak_memory
    peak_memory = 0
    stop_event = threading.Event()
    memory_thread = threading.Thread(target=track_memory, args=(stop_event,))
    memory_thread.start()

    query_index = build_index(query_embeddings, query_int_ids, "Flat")
    doc_index = build_index(doc_embeddings, doc_int_ids, index_type)

    faiss.write_index(query_index, f"{index_type}_{N_LIST}_query.index")
    faiss.write_index(doc_index, f"{index_type}_{N_LIST}_doc.index")

    stop_event.set()
    memory_thread.join()

    index_build_peak = peak_memory
    return index_build_peak

index_build_peak = build_and_save_indexes(args.index_type)
end_time = time.time()
print(f"{args.index_type} Index Time: {end_time - start_time} seconds")
print(f"{args.index_type} Index Memory: {index_build_peak} MB")

