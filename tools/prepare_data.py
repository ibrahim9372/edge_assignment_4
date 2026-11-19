"""
Fallback dataset preparation (used when the official CIFAR server is unreachable).

Converts the Hugging Face mirrors of CIFAR-10 / CIFAR-100 (parquet files with PNG-encoded
images) into compact NumPy archives  data/cifar10.npz  and  data/cifar100.npz  holding
x_train/x_test (uint8 NHWC), y_train/y_test and the class names.
`common.data.load_cifar` uses these archives when present, otherwise torchvision.

Download (any way you like) into data/hf/:
  https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/{train,test}-00000-of-00001.parquet
  https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{train,test}-00000-of-00001.parquet
saved as c100_train.parquet, c100_test.parquet, c10_train.parquet, c10_test.parquet, then:

    python -m tools.prepare_data
"""
import io
import json

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def read_split(path, label_col):
    t = pq.read_table(path)
    names = json.loads(t.schema.metadata[b"huggingface"])["info"]["features"][label_col]["names"]
    imgs = t.column("img").to_pylist()
    x = np.stack([np.asarray(Image.open(io.BytesIO(d["bytes"])).convert("RGB"), np.uint8) for d in imgs])
    y = np.asarray(t.column(label_col).to_pylist(), np.int64)
    return x, y, names


def convert(prefix, out, label_col):
    xtr, ytr, names = read_split(f"data/hf/{prefix}_train.parquet", label_col)
    xte, yte, _ = read_split(f"data/hf/{prefix}_test.parquet", label_col)
    assert xtr.shape == (50000, 32, 32, 3) and xte.shape == (10000, 32, 32, 3)
    np.savez(out, x_train=xtr, y_train=ytr, x_test=xte, y_test=yte, classes=np.array(names))
    print(f"{out}: train {xtr.shape}, test {xte.shape}, {len(names)} classes, e.g. {names[:5]}")


if __name__ == "__main__":
    convert("c10", "data/cifar10.npz", "label")
    convert("c100", "data/cifar100.npz", "fine_label")
