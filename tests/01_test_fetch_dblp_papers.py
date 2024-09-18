from fetcher.dblp import fetch_dblp_papers
from utils.common import (
    add_google_sheet,
    get_current_datetime
)

urls = [
    "https://dblp.org/db/conf/acl/acl2024-1.html",
    "https://dblp.org/db/conf/acl/acl2024f.html",
    "https://dblp.org/db/conf/naacl/naacl2024.html",
    "https://dblp.org/db/conf/naacl/naacl2024f.html",
    "https://dblp.org/db/conf/naacl/naacl2024-2.html",
    "https://dblp.org/db/conf/naacl/naacl2024-3.html",
    "https://dblp.org/db/conf/naacl/naacl2024-4.html",
    "https://dblp.org/db/conf/naacl/naacl2024-6.html",
    "https://dblp.org/db/conf/emnlp/emnlp2023.html",
    "https://dblp.org/db/conf/emnlp/emnlp2023f.html",
    "https://dblp.org/db/conf/emnlp/emnlp2023d.html",
    "https://dblp.org/db/conf/emnlp/emnlp2023i.html"
]

df = fetch_dblp_papers(urls)
add_google_sheet(df, "Analytics", get_current_datetime())