import duckdb
import pandas as pd

from setting import setting

DATA_PATH = setting.DATASET_PATH / "arxiv-metadata-oai-snapshot.json"
DB_PATH = setting.DATASET_PATH / "arxiv.db"

df = pd.read_json(DATA_PATH, lines=True, orient="records")

con = duckdb.connect(DB_PATH)
if "id_map" in con.execute("SHOW TABLES").fetchall():
    id_map = con.execute("SHOW * FOR id_map").fetchall()
    known_ids = set(id_map["arxiv_id"])
    max_int_id = id_map["int_id"].max()
else:
    id_map = pd.DataFrame(columns=["arxiv_id", "int_id"])
    known_ids = set()
    max_int_id = -1

new_entries = df[~df["id"].isin(known_ids)].copy()
new_entries["int_id"] = range(max_int_id + 1, max_int_id + 1 + len(new_entries))

new_mapping = pd.concat([
    id_map,
    new_entries[["id", "int_id"]].rename(columns={"id": "arxiv_id"})
], ignore_index=True)
df = df.merge(new_mapping, left_on="id", right_on="arxiv_id", how="left")

con.execute("DROP TABLE IF EXISTS arxiv;")
con.register("df", df)
con.execute("CREATE TABLE arxiv AS SELECT * FROM df;")

con.execute("DROP TABLE IF EXISTS id_map;")
con.register("new_mapping", new_mapping)
con.execute("CREATE TABLE id_map AS SELECT * FROM new_mapping;")