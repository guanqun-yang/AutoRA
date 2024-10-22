import bm25s
import Stemmer
import pandas as pd

from setting import setting

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


df = pd.read_json(
    setting.DATASET_PATH / "2015_2024_usenix_icse.json",
    lines=True,
    orient="records"
)

documents = df.title.tolist()
index = create_index(documents)

query = "explain llm language model security detection commit"
search_df = search_index(query, documents, index, k=10)
result_df = pd.merge(left=df, right=search_df, left_on="title", right_on="text")



