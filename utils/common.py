import time
import string
import random
import pickle
import gspread
import pathlib

import pandas as pd

from nltk import corpus
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

from setting import setting

def get_system_time():
    return str(time.time()).split(".")[0]


def get_current_datetime():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def generate_random_string(k):
    return "".join(random.sample(string.punctuation + string.ascii_letters + " ", k))


def generate_random_sentence():
    lst = [" ".join(tokens) for tokens in corpus.gutenberg.sents('shakespeare-macbeth.txt') if len(tokens) >= 10]
    return random.choice(lst)


def set_pandas_display(max_colwidth=100):
    pd.options.display.max_rows = None
    pd.options.display.max_colwidth = max_colwidth
    pd.options.display.max_columns = None


def launch_google_sheet_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive"
    ]

    creds = ServiceAccountCredentials.from_json_keyfile_name(
        setting.ROOT_DIRECTORY / pathlib.Path("utils/credential.json"),
        scopes=scopes
    )
    client = gspread.authorize(creds)

    return client


def save_pickle_file(data, file_name):
    with open(file_name, "wb") as fp:
        pickle.dump(data, fp)


def load_pickle_file(file_name):
    with open(file_name, "rb") as fp:
        return pickle.load(fp)


def add_google_sheet(df, filename, sheetname):
    client = launch_google_sheet_client().open(filename)
    client.add_worksheet(sheetname, rows=len(df), cols=len(df.columns)) \
          .update([df.columns.values.tolist()] + df.values.tolist())


def load_google_sheet(filename, sheetname):
    client = launch_google_sheet_client().open(filename)
    df = pd.DataFrame(client.worksheet(sheetname).get_all_records())

    return df