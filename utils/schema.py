import pandas as pd
from datetime import datetime

ARXIV_CATEGORIES = [
    "cs.CL", "cs.SE", "cs.CR", "cs.LG", "cs.AI", "cs.IR",
]
def extract_year(date_string):
    try:
        year = datetime.striptime(date_string, "%Y-%m-%d").year
    except Exception:
        year = 1900

    return year


def create_categories(row):
    shared_categories = set.intersection(
        set(ARXIV_CATEGORIES),
        set([category.strip() for category in row["categories"].split(",")])
    )

    for category in ARXIV_CATEGORIES:
        if category in shared_categories:
            row[category] = 1
        else:
            row[category] = 0

    return row


def align_dataset_schema(df, dataset_name):
    # cs.CL, cs.SE, cs.CR, cs.LG, cs.AI, cs.IR
    # make sure each dataset have the following consistent names, they could have other columns
    # - title (string)
    # - abstract (string)
    # - authors (string)
    # - url (string)
    # - year (int): this should be in the format of YYYY
    # - acl (boolean)
    # - cs.LG (boolean)
    # - cs.CL (boolean)
    # - cs.AI (boolean)
    # - cs.IR (boolean)
    # - cs.SE (boolean)
    # - cs.CR (boolean)

    assert dataset_name in ["acl", "arxiv"]

    if dataset_name == "acl":
        # original acl columns:
        # url, publisher, address, year, month, editor, title, ENTRYTYPE, ID, abstract, pages, doi, booktitle, author
        # volume, journal, language, isbn, note

        df = df[["title", "abstract", "author", "url", "year"]]
        df = df.rename(columns={"author": "authors"})

        # create multi-hot categories
        df["acl"] = 1
        for category in ARXIV_CATEGORIES:
            df[category] = 0
    else:
        # original arxiv columns:
        # string:
        # - "id", "submitter", "authors", "title", "comments",
        # - "journal-ref", "doi", "report-no", "categories", "license",
        # - "abstract", "update_date"
        # list:
        # - "versions", "authors_parsed"

        df = df[["title", "abstract", "authors", "update_date", "id", "categories"]]

        # custom title
        df["title"] = df.apply(lambda x: "[{}] {}".format(x["id"], x["title"]), axis="columns")

        df["update_date"] = df.update_date.apply(extract_year)
        df["id"] = df["id"].apply(lambda x: f"https://www.arxiv.org/abs/{x}")

        df = df.rename(columns={"update_date": "year", "id": "url"})

        # create multi-hot category
        df["acl"] = 0
        df = df.apply(create_categories, axis="columns")

    return df