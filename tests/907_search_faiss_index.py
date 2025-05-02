import torch
import faiss
import duckdb
import argparse

import numpy as np
import pandas as pd

from setting import setting
from datetime import datetime, timedelta
from utils.embedder import (
    get_hf_embedding,
    get_model_and_tokenizer,
)
from utils.common import (
    add_google_sheet,
    add_github_page,
    get_current_datetime,
)

##################################################

DB_PATH = setting.DATASET_PATH / "arxiv.db"
FAISS_PATH = setting.DATASET_PATH / "arxiv.faiss"
EMBEDDING_MODEL = "intfloat/e5-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
TOP_K = 200

model, tokenizer = get_model_and_tokenizer(EMBEDDING_MODEL, DEVICE)
##################################################

def format_markdown_page(df):
    lines = list()

    for idx, row in df.iterrows():
        rank = idx + 1

        title = row["title"]
        authors = row["authors"]
        abstract = row["abstract"]
        arxiv_id = row["arxiv_id"]
        update_date = row["update_date"]

        # Build PDF link
        pdf_link = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        # Build formatted block
        block = f"**[{rank}. [{arxiv_id}] {title}]({pdf_link})** ({update_date})\n\n" \
                f"*{authors}*\n\n" \
                f"{abstract}\n\n" \
                "---\n"

        lines.append(block)

    return "\n".join(lines)


##################################################

parser = argparse.ArgumentParser()
parser.add_argument("--search_query", type=str, default="bert")
args = parser.parse_args()

SEARCH_QUERY = args.search_query

##################################################

doc_index = faiss.read_index(str(FAISS_PATH))
query_vecs = np.array(get_hf_embedding(model, tokenizer, [SEARCH_QUERY]))

_, indices = doc_index.search(query_vecs, TOP_K)

con = duckdb.connect(database=str(DB_PATH), read_only=True)
all_indices = indices.flatten().tolist()

##################################################

placeholders = ",".join(["?"] * len(all_indices))
query = f"SELECT * FROM arxiv WHERE int_id IN ({placeholders})"

##################################################

df = con.execute(query, all_indices).df()

# condition 1
df['update_date'] = pd.to_datetime(df['update_date'], errors='coerce')
mask_date = (df['update_date'] >= datetime.now() - timedelta(days=5*365))

# condition 2
mask_cs = df['categories'].apply(lambda x: any(cat.startswith('cs.') for cat in x.split() if isinstance(x, str)))

df = df[mask_cs & mask_date]
df['update_date'] = df['update_date'].dt.strftime('%Y-%m-%d')

##################################################

df = df.set_index("int_id").loc[[int(idx) for idx in all_indices if int(idx) in df.int_id.tolist()]].reset_index()

##################################################

df["rank"] = df.index.tolist()
df["authors"] = df.authors_parsed.apply(lambda x: ", ".join([" ".join(xx[::-1]).strip() for xx in x]))
df["url"] = df.arxiv_id.apply(lambda x: "https://arxiv.org/abs/{}".format(x))

df = df[["rank", "arxiv_id", "update_date", "title", "abstract", "authors", "url"]]
markdown_string = format_markdown_page(df)

add_github_page(SEARCH_QUERY, markdown_string)

##################################################


# add_google_sheet(df.astype(str), "LiteratureAnalytics", get_current_datetime())
# print("✅ Google Sheet Updated")