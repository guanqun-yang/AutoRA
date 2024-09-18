import re
import requests

import pandas as pd

from bs4 import BeautifulSoup
from tqdm import tqdm


def fetch_dblp_papers(urls):
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

    titles = list()
    records = list()

    with tqdm(total=len(record_df)) as pbar:
        for _, row in record_df.iterrows():
            url = row["url"]

            response = requests.get(url)
            soup = BeautifulSoup(response.text, "lxml")
            titles.append(soup.title.text.strip("dblp:").strip())

            for raw_record in soup.find_all(class_="data tts-content"):
                records.append(
                    {
                        "year": row["year"],
                        "venue": row["venue"],
                        "title": raw_record.find(class_="title").text,
                        "authors": ", ".join([c.text for c in raw_record.find_all(itemprop="author")]),
                    }
                )

            pbar.update(1)

    df = pd.DataFrame(records)
    return df
