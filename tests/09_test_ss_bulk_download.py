import requests
import pandas as pd

from urllib.parse import urlencode


# https://api.semanticscholar.org/api-docs/

base_url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
params = {
    "query": "vulnerability assessment",
    "sort": "citationCount:desc",
    "year": "2014-2024",
    "minCitationCount": "20",
}

url = f"{base_url}?{urlencode(params)}"
response = requests.get(url)




