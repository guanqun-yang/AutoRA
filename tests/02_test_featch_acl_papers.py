import pandas as pd

from fetcher.acl import parse_large_bibtex_in_chunks
from setting import setting

entries = list()
for chunk in parse_large_bibtex_in_chunks(setting.DATASET_PATH / "anthology+abstracts.bib"):
    entries.extend(chunk)

    print(len(entries))

df = pd.DataFrame(entries)
df.to_pickle(setting.DATASET_PATH / "acl.pkl")