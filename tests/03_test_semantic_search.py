import faiss

import pandas as pd

from search.sentence_transformers import (
    embed_documents,
    create_index,
    search,
)
from utils.common import (
    add_google_sheet,
    load_google_sheet,
    get_current_datetime,
    save_pickle_file,
    load_pickle_file,
)
from setting import setting

df = pd.read_pickle(setting.DATASET_PATH / "acl.pkl")
df = df[df.year.isin(["2023", "2024"])]
records = df.to_dict(orient="records")

queries = [
    "explaination",
    "explainability",
    "reliability",
    "Large Language Model",
    "verify",
    "faithfulness",
    "rationale",
]


# embeddings = embed_documents(records, batch_size=16)
#
# save_pickle_file(embeddings, "acl_embedding.pkl")
embeddings = load_pickle_file("acl_embedding.pkl")

gpu_index = create_index(embeddings, batch_size=16)

# save index
cpu_index = faiss.index_gpu_to_cpu(gpu_index)
faiss.write_index(cpu_index, 'faiss_index.bin')

# load index
# cpu_index = faiss.read_index('faiss_index.bin')
# gpu_index= faiss.index_cpu_to_gpu(
#     faiss.StandardGpuResources(),
#     0,
#     cpu_index
# )

searched_df = search(queries, records, gpu_index, top_k=200)
add_google_sheet(
    searched_df.astype(str),
    "Search",
    get_current_datetime(),
)

