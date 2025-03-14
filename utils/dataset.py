import os
import os.path as osp
from PIL import Image
import six
import lmdb
import pickle
import numpy as np

import torch.utils.data as data
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)


def expand_greyscale(t):
    return t.expand(3, -1, -1)


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, image_size):
        self.db_path = db_path
        self.env = lmdb.open(
            db_path, 
            subdir=osp.isdir(db_path),
            readonly=True, 
            lock=False,
            readahead=False, 
            meminit=False
        )
        
        with self.env.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale)
        ])


    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_data(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return self.transform(img)

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_data(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pickle.dumps(obj)


def folder2lmdb(dpath, spath, write_frequency=50000):
    print("Loading dataset from %s" % dpath)
    dataset = ImageFolder(dpath, loader=raw_reader)
    data_loader = DataLoader(dataset, num_workers=16, collate_fn=lambda x: x)

    lmdb_path = osp.join(spath, "data.lmdb")
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(
        lmdb_path, 
        subdir=isdir,
        map_size=1099511627776 * 2, 
        readonly=False,
        meminit=False, 
        map_async=True
    )

    txn = db.begin(write=True)
    for idx, data in enumerate(data_loader):
        image, label = data[0]

        txn.put(u'{}'.format(idx).encode('ascii'), dumps_data((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_data(keys))
        txn.put(b'__len__', dumps_data(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()