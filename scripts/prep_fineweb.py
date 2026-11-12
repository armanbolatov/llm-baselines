"""Tokenize FineWeb sample-10BT into train.bin / val.bin with the same token stream as
src/data/fineweb.py (same split seed 2357 and test_size 1e-4, GPT-2 encode_ordinary + EOT, rows
in dataset order), without datasets.map: workers tokenize contiguous row ranges and return
uint16 arrays, written in order. No int64 token cache (src/data/fineweb.py's map step writes
~88 GB of it). Crash-safe: val.bin first, each file under .tmp, renamed when complete, READY last.

  python scripts/prep_fineweb.py <datasets_dir> <num_proc> [<parquet_dir>]

With <parquet_dir> (the sample/10BT/*.parquet files of HuggingFaceFW/fineweb, e.g. from
snapshot_download) the dataset is built from those files in sorted order, which is the order the
hub config "sample-10BT" uses; otherwise it is loaded by name.
"""
import glob
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import tiktoken
from datasets import load_dataset

DSET = None
ENC = None


def init(dset):
    global DSET, ENC
    DSET = dset
    ENC = tiktoken.get_encoding("gpt2")


def tokenize_range(bounds):
    lo, hi = bounds
    texts = DSET[lo:hi]["text"]
    out = []
    for ids in ENC.encode_ordinary_batch(texts, num_threads=1):
        ids.append(ENC.eot_token)
        out.append(np.asarray(ids, dtype=np.uint16))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.uint16)


def main():
    datasets_dir, num_proc = sys.argv[1], int(sys.argv[2])
    parquet_dir = sys.argv[3] if len(sys.argv) > 3 else None
    sample = os.environ.get("FINEWEB_SAMPLE", "sample-10BT")
    out = os.path.join(datasets_dir, f"fineweb-{sample.split('-')[-1]}")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    if parquet_dir:
        files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        print(f"{len(files)} parquet files from {parquet_dir}", flush=True)
        dataset = load_dataset("parquet", data_files=files, split="train", num_proc=min(num_proc, len(files)))
    else:
        dataset = load_dataset("HuggingFaceFW/fineweb", name=sample, split="train",
                               streaming=False, verification_mode="no_checks", num_proc=num_proc)
    print(f"loaded {len(dataset)} rows in {(time.time()-t0)/60:.1f} min", flush=True)
    split = dataset.train_test_split(test_size=0.0001, seed=2357, shuffle=True)
    split["val"] = split.pop("test")
    for name in ("val", "train"):
        final = os.path.join(out, f"{name}.bin")
        if os.path.exists(final):
            print(f"{final} exists, kept", flush=True)
            continue
        dset = split[name].select_columns(["text"])
        n = len(dset)
        step = 2000
        ranges = [(i, min(i + step, n)) for i in range(0, n, step)]
        tmp = final + ".tmp"
        total = 0
        with open(tmp, "wb") as f, Pool(num_proc, initializer=init, initargs=(dset,)) as pool:
            for k, arr in enumerate(pool.imap(tokenize_range, ranges, chunksize=1)):
                f.write(arr.tobytes())
                total += len(arr)
                if k % 500 == 0:
                    print(f"{name}: {k}/{len(ranges)} ranges, {total} tokens, "
                          f"{(time.time()-t0)/60:.1f} min", flush=True)
        assert os.path.getsize(tmp) == 2 * total
        os.replace(tmp, final)
        print(f"{name}.bin written: {total} tokens", flush=True)
    open(os.path.join(out, "READY"), "w").write(f"{time.ctime()}\n")
    print("FINEWEB_PREP_DONE", f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
