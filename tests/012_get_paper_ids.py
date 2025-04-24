import requests

import pandas as pd

from bs4 import BeautifulSoup

url = "https://dblp.org/db/conf/emnlp/emnlp2024.html"
response = requests.get(url)

soup = BeautifulSoup(response.text, "lxml")
