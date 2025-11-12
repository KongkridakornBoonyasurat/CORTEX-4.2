# C:\mnist\mnist_check.py  (no extra packages needed)
import struct, gzip, os, sys

ROOT = r"C:\Users\User\Desktop\Brain AI\cortex 4.2\cortex 4.2 v42"

def _open(path):
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")

def find_file(basenames):
    # try without .gz first, then with .gz
    for name in basenames:
        p1 = os.path.join(ROOT, name)
        p2 = p1 + ".gz"
        if os.path.exists(p1): return p1
        if os.path.exists(p2): return p2
    sys.exit(f"Missing file: one of {basenames}(.gz) not found in {ROOT}")

def read_images(path):
    with _open(path) as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic != 2051:
            sys.exit(f"{os.path.basename(path)}: bad magic {magic} (expected 2051)")
        n, rows, cols = struct.unpack(">III", f.read(12))
        data = f.read(n*rows*cols)
        if len(data) != n*rows*cols:
            sys.exit("images file truncated")
        # return first image only (to keep this fast)
        first = list(data[:rows*cols])
        return n, rows, cols, first

def read_labels(path):
    with _open(path) as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic != 2049:
            sys.exit(f"{os.path.basename(path)}: bad magic {magic} (expected 2049)")
        n = struct.unpack(">I", f.read(4))[0]
        labels = f.read(n)
        if len(labels) != n:
            sys.exit("labels file truncated")
        return n, labels[0]

def main():
    img_train = find_file(["train-images-idx3-ubyte"])
    lab_train = find_file(["train-labels-idx1-ubyte"])
    img_test  = find_file(["t10k-images-idx3-ubyte"])
    lab_test  = find_file(["t10k-labels-idx1-ubyte"])

    n_tr_i, r, c, first_img = read_images(img_train)
    n_tr_l, first_lab = read_labels(lab_train)
    n_te_i, _, _, _ = read_images(img_test)
    n_te_l, _ = read_labels(lab_test)

    ok = (n_tr_i==60000 and n_tr_l==60000 and n_te_i==10000 and n_te_l==10000)
    print("MNIST FOUND:", "OK" if ok else "UNEXPECTED SIZES")
    print(f"train images: {n_tr_i}  train labels: {n_tr_l}")
    print(f"test  images: {n_te_i}  test  labels: {n_te_l}")
    print(f"image size: {r}x{c}")
    print(f"first train label: {first_lab}")
    ink = sum(1 for v in first_img if v>0)
    print(f"first image nonzero pixels: {ink}/{r*c}")

if __name__ == "__main__":
    main()
