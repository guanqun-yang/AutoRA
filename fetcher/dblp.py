import re
import requests

import pandas as pd

from bs4 import BeautifulSoup
from tqdm import tqdm

from utils.common import convert_text_to_hash


def fetch_dblp_papers(urls, verbose=False):
    records = list()
    for url in urls:
        year_search_result = re.search("\d{4}", url)
        year = int(year_search_result[0])

        venue_search_result = re.search(r'db/conf/([^/]+)/', url)
        venue = venue_search_result.group(1).upper() if venue_search_result else "UNKNOWN"

        records.append(
            {
                "year": year,
                "url": url,
                "venue": venue,
            }
        )

    record_df = pd.DataFrame(records)
    records = list()

    with tqdm(total=len(record_df), disable=not verbose) as pbar:
        for _, row in record_df.iterrows():
            url = row["url"]

            response = requests.get(url)
            soup = BeautifulSoup(response.text, "lxml")

            # for raw_record in soup.find_all(class_="data tts-content"):
            #     records.append(
            #         {
            #             "dblp_id": None,
            #             "year": row["year"],
            #             "venue": row["venue"],
            #             "title": raw_record.find(class_="title").text,
            #             "authors": ", ".join([c.text for c in raw_record.find_all(itemprop="author")]),
            #         }
            #     )

            for raw_record in soup.find_all(class_="entry inproceedings"):
                records.append(
                    {
                        **{"year": row["year"], "venue": row["venue"]},
                        **process_dblp_entry(raw_record)
                    }
                )
            pbar.update(1)

    df = pd.DataFrame(records)
    return df



def process_dblp_entry(entry):
    """
    This function operate on one entry of the DBLP list. Such list of DBLP entries will returned by
    ```python
    import requests
    from bs4 import BeautifulSoup

    response = requests.get("https://dblp.org/db/conf/emnlp/emnlp2024.html")
    soup = BeautifulSoup(response.text, "lxml")
    entries = soup.find_all(class_="entry inproceedings"):
    ```
    It will return the extracted information in a dictionary.
    """
    d = dict()

    # title
    title_element = entry.find("span", class_="title")
    d["title"] = title_element.text.strip() if title_element else ""

    # id
    d["dblp_id"] = entry.get("id", convert_text_to_hash(d["title"]))

    # author
    authors = entry.find_all("span", itemprop="name")
    d["authors"] = ", ".join(author.text.strip() for author in authors)

    return d