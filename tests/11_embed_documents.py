import time
import pandas as pd

from tqdm import tqdm
from langchain_voyageai import VoyageAIEmbeddings
from utils.common import set_env

from setting import setting

set_env()
tqdm.pandas()


df = pd.read_json(
    "2025-01-26-23-45-34.json",
    lines=True,
    orient="records",
)

embedder = VoyageAIEmbeddings(
    model="voyage-3"
)

def embed(text):
    # rate limit: 2000 request per minute
    # src: https://docs.voyageai.com/docs/rate-limits
    time.sleep(0.001)
    return embedder.embed_query(text)

df["title_embedding"] = df.title.progress_apply(lambda x: embed(x))
df.to_pickle(setting.DATASET_PATH / "embedding.pkl")
