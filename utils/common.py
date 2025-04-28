import os
import base64
import yaml
import time
import string
import random
import pickle
import gspread
import pathlib
import hashlib
import requests
import itertools

import pandas as pd

from nltk import corpus
from functools import reduce
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


def get_env():
    with open(setting.ROOT_DIRECTORY / "env.yaml", "r") as fp:
        content = yaml.safe_load(fp)

    d = reduce(
        lambda x, y: {**x, **y},
        list(itertools.chain(*itertools.chain(content.values())))
    )

    return d


def set_env():
    d = get_env()

    for key, val in d.items():
        os.environ[key] = str(val)


def convert_text_to_hash(text):
    return  hashlib.sha256(text.encode()).hexdigest()


def add_github_page(query, content):
    set_env()

    REPO_OWNER = "guanqun-yang"
    REPO_NAME = "literature-analytics"
    GITHUB_TOKEN = os.environ["PAT"]

    session = requests.Session()
    session.headers.update({
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    })

    now = datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    timestamp = now.strftime("%Y%m%d")
    safe_query_name = query.replace(" ", "_").lower()

    encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
    new_md_path = f"{year}/{month}/{timestamp}_{safe_query_name}.md"

    upload_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{new_md_path}"

    # check if file the already uploaded
    r = session.get(upload_url)
    if r.status_code == 200:
        sha = r.json()["sha"]
    else:
        sha = None

    payload = {
        "message": f"Upload {new_md_path}",
        "content": encoded_content,
        "branch": "main"
    }
    if sha:
        payload["sha"] = sha

    r = session.put(upload_url, json=payload)
    if r.status_code in [201, 200]:
        print(f"✅ Successfully uploaded {new_md_path}")
    else:
        print(f"❌ Failed to upload {new_md_path}: {r.json()}")

    ##################################################
    # UPDATE README.md
    ##################################################
    readme_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/README.md"
    r = session.get(readme_url)

    if r.status_code == 200:
        readme_data = r.json()
        current_readme = base64.b64decode(readme_data['content']).decode('utf-8')
        readme_sha = readme_data['sha']
    else:
        current_readme = "# Search Results\n\n"
        readme_sha = None

    # update
    new_link = f"- [{query.title()} ({timestamp})]({new_md_path})\n"
    updated_readme = current_readme + new_link

    # upload
    payload = {
        "message": "Update README.md with new search result",
        "content": base64.b64encode(updated_readme.encode('utf-8')).decode('utf-8'),
        "branch": "main"
    }
    if readme_sha:
        payload["sha"] = readme_sha

    r = session.put(readme_url, json=payload)
    if r.status_code in [201, 200]:
        print(f"✅ Successfully updated README.md")
    else:
        print(f"❌ Failed to update README.md: {r.json()}")