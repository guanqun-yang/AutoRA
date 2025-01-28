import re
import requests

import pandas as pd

from bs4 import BeautifulSoup
from wordcloud import WordCloud

import matplotlib.pyplot as plt

url = "https://dblp.org/pid/140/7129.html"

response = requests.get(url)
soup = BeautifulSoup(response.content, "lxml")

publication_entries = soup.find_all('li', attrs={'itemtype': 'http://schema.org/ScholarlyArticle'})

records = list()

for entry in publication_entries:
    title = entry.find(attrs={"class": "title"}).get_text(strip=True)
    authors = [
        author.get_text(strip=True)
        for author in entry.find_all('span', itemprop='name', recursive=True)
    ]
    year = entry.find('span', itemprop='datePublished').get_text(strip=True)
    venue = entry.find('span', itemprop='isPartOf').find('span', itemprop='name').get_text(strip=True)

    records.append(
        {
            "title": title,
            "author": ", ".join(authors),
            "venue": venue,
            "year": year,
        }
    )

df = pd.DataFrame(records)

aggregated_text = " ".join(df.title.tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(aggregated_text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()