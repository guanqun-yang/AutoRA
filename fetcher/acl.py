import os
import wget
import pandas as pd
import bibtexparser

from tqdm import trange
from setting import setting


def parse_large_bibtex_in_chunks(file_path, chunk_size=1000):
    def stream_bibtex_entries(file_obj):
        entry_lines = []
        in_entry = False

        for line in file_obj:
            stripped_line = line.strip()
            if stripped_line.startswith('@'):
                if in_entry:
                    # yield the last entry before starting the new one
                    yield ''.join(entry_lines)
                    entry_lines = []
                in_entry = True

            if in_entry:
                entry_lines.append(line)

        # yield the last entry
        if entry_lines:
            yield ''.join(entry_lines)

    with open(file_path, 'r', encoding='utf-8') as bibtex_file:
        entry_gen = stream_bibtex_entries(bibtex_file)
        entries = []

        for entry_str in entry_gen:
            entry = bibtexparser.loads(entry_str).entries[0]  # Parse single entry
            entries.append(entry)

            if len(entries) >= chunk_size:
                yield entries
                entries = []

        # yield remaining entries
        if entries:
            yield entries


