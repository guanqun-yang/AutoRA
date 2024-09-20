from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import sys
import pathlib

common_packages = [
    "nltk",
    "numpy",
    "pandas",
    "termcolor",
    "scikit-learn",
    "seaborn",
    "beautifulsoup4",
    "gspread",
    "oauth2client",
    "lxml",
    "wget",
    "bibtexparser",
    "flask",
    "sentence_transformers",
    "faiss-gpu",
]

setup(
    name='autora',
    version='0.1',
    packages=find_packages(),
    install_requires=common_packages,
)