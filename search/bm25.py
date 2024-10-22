import bm25s
import Stemmer

import pandas as pd
from setting import setting

# pip install bm25s
# pip install PyStemmer

STEMMER = Stemmer.Stemmer("english")

def create_index(documents):
    """

    :param documents: a list of strings
    :return: the handler to crated index
    """
    tokenized_documents = bm25s.tokenize(documents, stopwords="en", stemmer=STEMMER)

    index = bm25s.BM25()
    index.index(tokenized_documents)

    return index

def search_index(query, documents, index, k):
    """

    :param query:
    :param documents:
    :param index: the index created in create_index()
    :param k: top-k
    :return:
    """

    tokenized_query = bm25s.tokenize(query, stemmer=STEMMER)
    results, scores = index.retrieve(
        tokenized_query,
        documents,
        k=k
    )

    return pd.DataFrame(
        [
            {
                "text": result,
                "score": score
            }
            for result, score in zip(results[0], scores[0])
        ]
    )
