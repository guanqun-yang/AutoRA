import itertools
import pandas as pd

from fetcher.dblp import fetch_dblp_papers
from utils.common import add_google_sheet, get_current_datetime
from setting import setting

START_YEAR = 2000
END_YEAR = 2024

usenix_urls = [
    f"https://dblp.org/db/conf/uss/uss{year}.html"
    for year in range(START_YEAR, END_YEAR + 1)
]

icse_urls = list(itertools.chain.from_iterable(
    [
        f"https://dblp.org/db/conf/icse/icse{year}.html",
        f"https://dblp.org/db/conf/icse/icse{year}c.html",
        f"https://dblp.org/db/conf/icse/nier{year}.html",
    ]
    for year in range(START_YEAR, END_YEAR + 1)
))

issta_urls = [
    f"https://dblp.org/db/conf/issta/issta{year}.html"
    for year in range(START_YEAR, END_YEAR + 1)
]

fse_urls = [
    f"https://dblp.org/db/conf/sigsoft/fse{year}.html"
    for year in range(START_YEAR, END_YEAR + 1)
]

ase_urls = [
    f"https://dblp.org/db/conf/kbse/ase{year}.html"
    for year in range(START_YEAR, END_YEAR + 1)
]

msr_urls = [
    f"https://dblp.org/db/conf/msr/msr{year}.html"
    for year in range(START_YEAR, END_YEAR + 1)
]

# additional_icse_urls = [
#     # 2016
#     "https://dblp.org/db/conf/icse/icse2016.html",
#     "https://dblp.org/db/conf/icse/icse2016c.html",
#     # 2015
#     "https://dblp.org/db/conf/icse/icse2015-1.html",
#     "https://dblp.org/db/conf/icse/icse2015-2.html",
# ]

urls = usenix_urls + icse_urls + issta_urls + fse_urls + ase_urls + msr_urls
df = fetch_dblp_papers(urls)
df.to_json(setting.DATASET_PATH / "se_2022_2014.json", lines=True, orient="records")

# add_google_sheet(
#     df,
#     "LiteratureAnalytics",
#     get_current_datetime()
# )