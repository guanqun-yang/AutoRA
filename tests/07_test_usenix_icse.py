import itertools
import pandas as pd

from fetcher.dblp import fetch_dblp_papers
from setting import setting

# usenix_urls = [
#     f"https://dblp.org/db/conf/uss/uss{year}.html"
#     for year in range(2014, 2025)
# ]
#
# icse_urls = list(itertools.chain.from_iterable(
#     [
#         f"https://dblp.org/db/conf/icse/icse{year}.html",
#         f"https://dblp.org/db/conf/icse/icse{year}c.html",
#         f"https://dblp.org/db/conf/icse/nier{year}.html",
#     ]
#     for year in range(2017, 2025)
# ))
#
#
# additional_icse_urls = [
#     # 2016
#     "https://dblp.org/db/conf/icse/icse2016.html",
#     "https://dblp.org/db/conf/icse/icse2016c.html",
#     # 2015
#     "https://dblp.org/db/conf/icse/icse2015-1.html",
#     "https://dblp.org/db/conf/icse/icse2015-2.html",
# ]
#
# urls = usenix_urls + icse_urls + additional_icse_urls
#
# df = fetch_dblp_papers(urls)


df = pd.read_json(setting.DATASET_PATH / "usenix_icse.json", lines=True, orient="records")
