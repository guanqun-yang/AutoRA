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
    "uss": ["https://dblp.org/db/conf/uss/uss{year}.html"],
    "icse": [
        "https://dblp.org/db/conf/icse/icse{year}.html",
        "https://dblp.org/db/conf/icse/icse{year}c.html",
        "https://dblp.org/db/conf/icse/nier{year}.html",
    ],
    "issta": ["https://dblp.org/db/conf/issta/issta{year}.html"],
    "msr": ["https://dblp.org/db/conf/msr/msr{year}.html"],
    "ase": ["https://dblp.org/db/conf/kbse/ase{year}.html"],
    "fse": ["https://dblp.org/db/conf/sigsoft/fse{year}.html"],
    "re": [
        "https://dblp.org/db/conf/re/re{year}.html",
        "https://dblp.org/db/conf/re/re{year}w.html"
    ],
    "issre": [
        "https://dblp.org/db/conf/issre/issre{year}.html",
        "https://dblp.org/db/conf/issre/issre{year}w.html"
    ],
    "saner": [
        "https://dblp.org/db/conf/wcre/saner{year}.html",

    ],
    "icsme": [
        "https://dblp.org/db/conf/icsm/icsme{year}.html",
    ],
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
}

years = range(START_YEAR, END_YEAR + 1)

# tracking progress
url_dict = {
    conf: [
        pattern.format(year=year)
        for year in years
        for pattern in patterns
    ]
    for conf, patterns in conference_patterns.items()
}
urls = list(itertools.chain(*url_dict.values()))

with tqdm(total=len(urls)) as pbar:
    for venue, patterns in conference_patterns.items():
        path = setting.DATASET_PATH / pathlib.Path(f"dblp/{venue}")
        if not path.exists():
            path.mkdir()

        for year in years:
            if (path / f"{year}.json").exists():
                continue

            urls = [pattern.format(year=year) for pattern in patterns]
            df = fetch_dblp_papers(urls)

            df.to_json(path / f"{year}.json", lines=True, orient="records")
            pbar.update(1)

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


# add_google_sheet(
#     df,
#     "LiteratureAnalytics",
#     get_current_datetime()
# )