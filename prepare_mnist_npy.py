import struct, numpy as np, pathlib
ROOT = pathlib.Path(r"9\MNIST")
def read_images(p):
    with open(p,'rb') as f:
        assert int.from_bytes(f.read(4),'big')==2051
        n=int.from_bytes(f.read(4),'big'); r=int.from_bytes(f.read(4),'big'); c=int.from_bytes(f.read(4),'big')
        return np.frombuffer(f.read(n*r*c), dtype=np.uint8).reshape(n,r,c)
def read_labels(p):
    with open(p,'rb') as f:
        assert int.from_bytes(f.read(4),'big')==2049
        n=int.from_bytes(f.read(4),'big')
        return np.frombuffer(f.read(n), dtype=np.uint8)
Xtr = read_images(ROOT/"train-images-idx3-ubyte"); Ytr = read_labels(ROOT/"train-labels-idx1-ubyte")
Xte = read_images(ROOT/"t10k-images-idx3-ubyte");  Yte = read_labels(ROOT/"t10k-labels-idx1-ubyte")
np.save(ROOT/"train_images.npy", Xtr); np.save(ROOT/"train_labels.npy", Ytr)
np.save(ROOT/"test_images.npy",  Xte); np.save(ROOT/"test_labels.npy",  Yte)
print("SAVED .npy:", Xtr.shape, Ytr.shape, Xte.shape, Yte.shape)
