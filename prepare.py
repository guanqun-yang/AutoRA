import time
import pathlib
import itertools
import pandas as pd

from fetcher.dblp import fetch_dblp_papers
from utils.common import add_google_sheet, get_current_datetime
from tqdm import tqdm
from setting import setting


START_YEAR = 2000
END_YEAR = 2024

conference_patterns = {
    # software engineering
    "icse": [
        "https://dblp.org/db/conf/icse/icse{year}.html",
        "https://dblp.org/db/conf/icse/icse{year}c.html",
        "https://dblp.org/db/conf/icse/nier{year}.html",
    ],
    "ase": ["https://dblp.org/db/conf/kbse/ase{year}.html"],
    "fse": ["https://dblp.org/db/conf/sigsoft/fse{year}.html"],
    "icsme": [
        "https://dblp.org/db/conf/icsm/icsme{year}.html",
    ],
    "saner": [
        "https://dblp.org/db/conf/wcre/saner{year}.html",
    ],
    "msr": ["https://dblp.org/db/conf/msr/msr{year}.html"],
    # testing
    "issta": ["https://dblp.org/db/conf/issta/issta{year}.html"],
    # requirement engineering
    "re": [
        "https://dblp.org/db/conf/re/re{year}.html",
        "https://dblp.org/db/conf/re/re{year}w.html"
    ],
    "issre": [
        "https://dblp.org/db/conf/issre/issre{year}.html",
        "https://dblp.org/db/conf/issre/issre{year}w.html"
    ],
    # hci
    "chi": [
        "https://dblp.org/db/conf/chi/chi{year}.html",
        "https://dblp.org/db/conf/chi/chi{year}w.html"
    ],
    "cscw": [
        "https://dblp.org/db/conf/cscw/cscw{year}c.html"
    ],
    "iui": [
        "https://dblp.org/db/conf/iui/iui{year}.html",
        "https://dblp.org/db/conf/iui/iui{year}c.html",
        "https://dblp.org/db/conf/iui/iui{year}w.html"
    ],
    # nlp
    "emnlp": [
        "https://dblp.org/db/conf/emnlp/emnlp{year}.html"
    ],
    "acl": [
        "https://dblp.org/db/conf/acl/acl{year}-1.html",
        "https://dblp.org/db/conf/acl/acl{year}-2.html"
    ],
    "naacl": [
        "https://dblp.org/db/conf/naacl/naacl{year}-1.html",
        "https://dblp.org/db/conf/naacl/naacl{year}-2.html"
    ],
    # security
    "dimva": [
        "https://dblp.org/db/conf/dimva/dimva{year}.html"
    ],
    "ndss": [
        "https://dblp.org/db/conf/ndss/ndss{year}.html"
    ],
    "uss": ["https://dblp.org/db/conf/uss/uss{year}.html"],
    "ccs": [
        "https://dblp.org/db/conf/ccs/ccs{year}.html",
    ],
    "sp": [
        "https://dblp.org/db/conf/sp/sp{year}.html",
        "https://dblp.org/db/conf/sp/sp{year}w.html"
    ]

}

years = range(START_YEAR, END_YEAR + 1)

for venue, patterns in tqdm(conference_patterns.items(), desc="Processing Conferences..."):
    path = setting.DATASET_PATH / pathlib.Path(f"dblp/{venue}")
    if not path.exists():
        path.mkdir()

    for year in years:
        filename = path / f"{year}.json"
        do_download = not filename.exists() or filename.stat().st_size / 1024 <= 5

        if do_download:
            urls = [pattern.format(year=year) for pattern in patterns]
            df = fetch_dblp_papers(urls)
            df.to_json(path / f"{year}.json", lines=True, orient="records")
            time.sleep(5)


df = pd.concat(
    [
        pd.read_json(filename, lines=True, orient="records")
        for filename in (setting.DATASET_PATH / "dblp").rglob("*.json")
    ]
).reset_index(drop=True)

df = df.to_json(
    setting.DATASET_PATH / "{}.json".format(get_current_datetime()),
    lines=True,
    orient="records"
)
