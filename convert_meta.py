import json
import os

metadir = "data/dataset_chart/real_meta"
meta_list = os.listdir(metadir)
for meta in meta_list:
    if not meta.endswith(".json"):
        continue

    metapath = os.path.join(metadir, meta)
    with open(metapath, "r") as f:
        meta = json.load(f)
    start, fps = meta["start"]
    end = meta["end"][0]
    meta["start"] = start
    meta["end"] = end
    meta["fps"] = fps
    with open(metapath, "w") as f:
        json.dump(meta, f)
