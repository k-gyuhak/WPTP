import numpy as np
import os
import torch

from torchvision import datasets, transforms

import copy

from args import args
import utils
from torch.utils.data.dataset import Subset

from data.custom_imagefolder import ImageFolder

# things for MNIST
from torchvision.datasets.vision import VisionDataset
import warnings
from PIL import Image
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
from torchvision.datasets.utils import download_url, download_and_extract_archive, extract_archive, \
    verify_str_arg

from data.mnist_mod import MNIST as MNIST_MOD

def get_int(b: bytes) -> int:
    return int(codecs.encode(b, "hex"), 16)


SN3_PASCALVINCENT_TYPEMAP = {
    8: torch.uint8,
    9: torch.int8,
    11: torch.int16,
    12: torch.int32,
    13: torch.float32,
    14: torch.float64,
}


def read_sn3_pascalvincent_tensor(path: str, strict: bool = True) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
    Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open(path, "rb") as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    torch_type = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]

    num_bytes_per_value = torch.iinfo(torch_type).bits // 8
    # The MNIST format uses the big endian byte order. If the system uses little endian byte order by default,
    # we need to reverse the bytes before we can read them with torch.frombuffer().
    needs_byte_reversal = sys.byteorder == "little" and num_bytes_per_value > 1
    parsed = torch.frombuffer(bytearray(data), dtype=torch_type, offset=(4 * (nd + 1)))
    if needs_byte_reversal:
        parsed = parsed.flip(0)

    assert parsed.shape[0] == np.prod(s) or not strict
    return parsed.view(*s)

def read_label_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    assert x.dtype == torch.uint8
    assert x.ndimension() == 1
    return x.long()


def read_image_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    assert x.dtype == torch.uint8
    assert x.ndimension() == 3
    return x

class MNIST(VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    resources = [
        ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")



    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

def partition_datasetv4_imgnet(dataset, perm):
    lperm = perm.tolist()
    newdataset = copy.copy(dataset)
    indices = [i for i, (_, label) in enumerate(newdataset.samples) if label in lperm]
    return indices

def partition_datasetv4(dataset, perm):
    lperm = perm.tolist()
    newdataset = copy.copy(dataset)
    newdataset.data = [
        im
        for im, label in zip(newdataset.data, newdataset.targets)
        if label in lperm
    ]

    newdataset.targets = [
        lperm.index(label)
        for label in newdataset.targets
        if label in lperm
    ]
    return newdataset

class SplitMNIST:
    def __init__(self):
        super(SplitMNIST, self).__init__()
        # data_root = os.path.join(args.data, "cifar100")
        num_cls = args.output_size
        # data_root = os.path.join(args.data, "MNIST")
        data_root = args.data

        use_cuda = torch.cuda.is_available()

        args.mean = (0.1307, 0.1307, 0.1307)
        args.std = (0.3081, 0.3081, 0.3081)
        normalize = transforms.Normalize(
            mean=(0.1307), std=(0.3081)
        )

        train_dataset = MNIST_MOD(
            root=args.data,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(28, padding=2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        val_dataset = MNIST_MOD(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )

        np.random.seed(args.seed)
        # perm = np.random.permutation(100)
        perm = np.arange(args.total_cls)
        print(perm)

        splits = [
            (
                partition_datasetv4(train_dataset, perm[num_cls * i:num_cls * (i+1)]),
                partition_datasetv4(val_dataset, perm[num_cls * i:num_cls * (i+1)]),
            )
            for i in range(args.num_tasks)
        ]

        [print(perm[num_cls * i:num_cls * (i+1)]) for i in range(args.num_tasks)]

        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        self.loaders = [
            (
                torch.utils.data.DataLoader(
                    x[0], batch_size=args.batch_size, shuffle=True, **kwargs
                ),
                torch.utils.data.DataLoader(
                    x[1], batch_size=args.test_batch_size, shuffle=True, **kwargs
                ),
            )
            for x in splits
        ]

    def update_task(self, i):
        self.train_loader = self.loaders[i][0]
        self.val_loader = self.loaders[i][1]

class RandSplitCIFAR10:
    def __init__(self):
        super(RandSplitCIFAR10, self).__init__()
        num_cls = args.output_size
        # data_root = os.path.join(args.data, "cifar100")
        data_root = args.data

        use_cuda = torch.cuda.is_available()

        args.mean = (0.491, 0.482, 0.447)
        args.std = (0.247, 0.243, 0.262)
        normalize = transforms.Normalize(
            mean=args.mean, std=args.std
        )

        train_dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        val_dataset = datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )

        np.random.seed(args.seed)
        perm = np.arange(args.total_cls)
        print(perm)

        splits = [
            (
                partition_datasetv4(train_dataset, perm[num_cls * i:num_cls * (i+1)]),
                partition_datasetv4(val_dataset, perm[num_cls * i:num_cls * (i+1)]),

            )
            for i in range(args.num_tasks)
        ]

        # for i in range(args.total_cls):
        #     print()
        #     print(f"=> Size of train split {i}: {len(splits[i][0].data)}")
        #     print(f"=> Size of val split {i}: {len(splits[i][1].data)}")
        [print(perm[num_cls * i:num_cls * (i+1)]) for i in range(args.num_tasks)]

        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        self.loaders = [
            (
                torch.utils.data.DataLoader(
                    x[0], batch_size=args.batch_size, shuffle=True, **kwargs
                ),
                torch.utils.data.DataLoader(
                    x[1], batch_size=args.test_batch_size, shuffle=True, **kwargs
                ),
            )
            for x in splits
        ]

    def update_task(self, i):
        self.train_loader = self.loaders[i][0]
        self.val_loader = self.loaders[i][1]

class RandSplitCIFAR100:
    def __init__(self):
        super(RandSplitCIFAR100, self).__init__()
        num_cls = args.output_size
        # data_root = os.path.join(args.data, "cifar100")
        data_root = args.data

        use_cuda = torch.cuda.is_available()

        args.mean = (0.491, 0.482, 0.447)
        args.std = (0.247, 0.243, 0.262)
        normalize = transforms.Normalize(
            mean=args.mean, std=args.std
        )

        train_dataset = datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        val_dataset = datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )

        np.random.seed(args.seed)
        # perm = np.random.permutation(100)
        perm = np.arange(args.total_cls)
        print(perm)

        splits = [
            (
                partition_datasetv4(train_dataset, perm[num_cls * i:num_cls * (i+1)]),
                partition_datasetv4(val_dataset, perm[num_cls * i:num_cls * (i+1)]),
            )
            for i in range(args.num_tasks)
        ]

        # for i in range(20):
        #     print(len(splits[i][0].data))
        #     print(len(splits[i][1].data))
        #     print("==")
        [print(perm[num_cls * i:num_cls * (i+1)]) for i in range(args.num_tasks)]

        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        self.loaders = [
            (
                torch.utils.data.DataLoader(
                    x[0], batch_size=args.batch_size, shuffle=True, **kwargs
                ),
                torch.utils.data.DataLoader(
                    x[1], batch_size=args.test_batch_size, shuffle=True, **kwargs
                ),
            )
            for x in splits
        ]

    def update_task(self, i):
        self.train_loader = self.loaders[i][0]
        self.val_loader = self.loaders[i][1]

class RandSplitTinyImg:
    def __init__(self):
        super(RandSplitTinyImg, self).__init__()
        # data_root = os.path.join(args.data, "cifar100")

        use_cuda = torch.cuda.is_available()

        args.mean = (0.485, 0.456, 0.406)
        args.std = (0.229, 0.224, 0.225)
        normalize = transforms.Normalize(
            mean=args.mean, std=args.std
        )

        train_dataset = ImageFolder(
            root=IMAGENET_PATH + '/train',
            transform=transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        val_dataset = ImageFolder(
            root=IMAGENET_PATH + '/val_folders',
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                normalize]),
        )

        np.random.seed(args.seed)
        # perm = np.random.permutation(100)
        perm = np.arange(args.total_cls)
        print(perm)

        splits = [
            (
                Subset(train_dataset, partition_datasetv4_imgnet(train_dataset, perm[args.output_size * i:args.output_size * (i+1)])),
                Subset(val_dataset, partition_datasetv4_imgnet(val_dataset, perm[args.output_size * i:args.output_size * (i+1)])),
                # partition_datasetv4(train_dataset, perm[20 * i:20 * (i+1)]),
                # partition_datasetv4(val_dataset, perm[20 * i:20 * (i+1)]),
            )
            for i in range(args.num_tasks)
        ]

        # for i in range(20):
        #     print(len(splits[i][0].data))
        #     print(len(splits[i][1].data))
        #     print("==")
        [print(perm[args.output_size * i:args.output_size * (i+1)]) for i in range(args.num_tasks)]

        for i in range(args.num_tasks):
            splits[i][0].dataset.targets = np.array(splits[i][0].dataset.targets) % args.output_size
            splits[i][1].dataset.targets = np.array(splits[i][1].dataset.targets) % args.output_size

        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        self.loaders = [
            (
                torch.utils.data.DataLoader(
                    x[0], batch_size=args.batch_size, shuffle=True, **kwargs
                ),
                torch.utils.data.DataLoader(
                    x[1], batch_size=args.test_batch_size, shuffle=True, **kwargs
                ),
            )
            for x in splits
        ]
        # for i, x in enumerate(splits):
        #     print(i, x[0])

    def update_task(self, i):
        self.train_loader = self.loaders[i][0]
        self.val_loader = self.loaders[i][1]


########################## CSI ##########################

def partition_datasetv4_imgnet_csi(dataset, perm, train=True, cal=False, val=False, prop=None):
    lperm = perm.tolist()
    newdataset = copy.copy(dataset)
    indices = []
    if train:
        train_idx_list, val_idx_list = [], []
        for label in perm:
            idx = np.where(np.array(dataset.targets) == label)[0]
            n_samples = len(idx)

            if prop:
                train_idx = idx[:int(n_samples * prop)]
                val_idx   = idx[int(n_samples * prop):]

                train_idx_list.append(train_idx)
                val_idx_list.append(val_idx)

            else:
                train_idx_list.append(idx)

            if cal:
                train_idx_list[-1] = train_idx_list[-1][-10:]
            else:
                train_idx_list[-1] = train_idx_list[-1][:-10]

        if val:
            final_idx = np.concatenate(val_idx_list).astype(int)
        else:
            final_idx = np.concatenate(train_idx_list).astype(int)

    else:
        final_idx = []
        for label in perm:
            idx = np.where(np.array(dataset.targets) == label)[0]
            final_idx.append(idx)
        final_idx = np.concatenate(final_idx)

        # if train and not cal:
        #     indices.append(np.where(np.array(dataset.targets) == label)[0][:-10])
        # elif train and cal:
        #     indices.append(np.where(np.array(dataset.targets) == label)[0][-10:])
        # elif not train and not cal:
        #     indices.append(np.where(np.array(dataset.targets) == label)[0])
        # else:
        #     raise NotImplementedError()
    # indices = np.concatenate(final_idx).tolist()
    return final_idx

def partition_datasetv4_csi(dataset, perm, train=True, cal=False, val=False, prop=None):
    if prop is not None:
        assert train is True
    if train is False:
        assert prop is None and val is False
    lperm = perm.tolist()
    newdataset = copy.copy(dataset)

    newdataset.targets = np.array(newdataset.targets)

    if train:
        train_idx_list, val_idx_list = [], []
        for c in lperm:
            idx = np.where(np.array(newdataset.targets) == c)[0]
            n_samples = len(idx)

            if prop:
                train_idx = idx[:int(n_samples * prop)]
                val_idx   = idx[int(n_samples * prop):]

                train_idx_list.append(train_idx)
                val_idx_list.append(val_idx)

            else:
                train_idx_list.append(idx)

            if cal:
                train_idx_list[-1] = train_idx_list[-1][-20:]
            else:
                train_idx_list[-1] = train_idx_list[-1][:-20]

        if val:
            final_idx = np.concatenate(val_idx_list).astype(int)
        else:
            final_idx = np.concatenate(train_idx_list).astype(int)

        newdataset.data = newdataset.data[final_idx]
        newdataset.targets = newdataset.targets[final_idx]
        return newdataset 

    else:
        newdataset.data = [
            im
            for im, label in zip(newdataset.data, newdataset.targets)
            if label in lperm
        ]

        newdataset.targets = [
            lperm.index(label)
            for label in newdataset.targets
            if label in lperm
        ]
        return newdataset

class SplitMNISTCSI:
    def __init__(self):
        super(SplitMNISTCSI, self).__init__()
        num_cls = args.output_size
        data_root = args.data

        use_cuda = torch.cuda.is_available()

        args.mean = (0.1307, 0.1307, 0.1307)
        args.std = (0.3081, 0.3081, 0.3081)
        normalize = transforms.Normalize(
            mean=(0.1307), std=(0.3081)
        )

        train_dataset = MNIST_MOD(
            root=args.data,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
        )
        val_dataset = MNIST_MOD(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )

        np.random.seed(args.seed)
        perm = np.arange(args.total_cls)
        print(perm)

        splits = [
            (
                partition_datasetv4_csi(train_dataset, perm[num_cls * i:num_cls * (i+1)], train=True, cal=False, val=False, prop=0.9) if args.validation \
                    else partition_datasetv4_csi(train_dataset, perm[num_cls * i:num_cls * (i+1)], train=True, cal=False),
                partition_datasetv4_csi(train_dataset, perm[num_cls * i:num_cls * (i+1)], train=True, cal=False, val=True, prop=0.9) if args.validation \
                    else partition_datasetv4_csi(val_dataset, perm[num_cls * i:num_cls * (i+1)], train=False, cal=False),
                partition_datasetv4_csi(train_dataset, perm[num_cls * i:num_cls * (i+1)], train=True, cal=True, val=False, prop=0.9) if args.validation \
                    else partition_datasetv4_csi(train_dataset, perm[num_cls * i:num_cls * (i+1)], train=True, cal=True),
            )
            for i in range(args.num_tasks)
        ]

        for i in range(args.num_tasks):
            splits[i][0].targets = np.array(splits[i][0].targets) % num_cls
            splits[i][1].targets = np.array(splits[i][1].targets) % num_cls
            splits[i][2].targets = np.array(splits[i][2].targets) % num_cls

        [print(perm[num_cls * i:num_cls * (i+1)]) for i in range(args.num_tasks)]

        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        self.loaders = [
            (
                torch.utils.data.DataLoader(
                    x[0], batch_size=args.batch_size, shuffle=True, **kwargs
                ),
                torch.utils.data.DataLoader(
                    x[1], batch_size=args.test_batch_size, shuffle=True, **kwargs
                ),
                torch.utils.data.DataLoader(
                    x[2], batch_size=args.cal_batch_size, shuffle=True, **kwargs
                ),
            )
            for x in splits
        ]

    def update_task(self, i):
        self.train_loader = self.loaders[i][0]
        self.val_loader = self.loaders[i][1]
        self.cal_loader = self.loaders[i][2]

class RandSplitCIFAR10CSI:
    def __init__(self):
        super(RandSplitCIFAR10CSI, self).__init__()
        data_root = args.data
        num_cls = args.output_size

        use_cuda = torch.cuda.is_available()

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        train_dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
        )
        val_dataset = datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                # normalize,
                ]),
        )

        np.random.seed(args.seed)
        # perm = np.random.permutation(100)
        perm = np.arange(args.total_cls)
        print(perm)

        splits = [
            (
                partition_datasetv4_csi(train_dataset, perm[num_cls * i:num_cls * (i+1)], train=True, cal=False, val=False, prop=0.9) if args.validation \
                    else partition_datasetv4_csi(train_dataset, perm[num_cls * i:num_cls * (i+1)], train=True, cal=False),
                partition_datasetv4_csi(train_dataset, perm[num_cls * i:num_cls * (i+1)], train=True, cal=False, val=True, prop=0.9) if args.validation \
                    else partition_datasetv4_csi(val_dataset, perm[num_cls * i:num_cls * (i+1)], train=False, cal=False),
                partition_datasetv4_csi(train_dataset, perm[num_cls * i:num_cls * (i+1)], train=True, cal=True, val=False, prop=0.9) if args.validation \
                    else partition_datasetv4_csi(train_dataset, perm[num_cls * i:num_cls * (i+1)], train=True, cal=True),
            )
            for i in range(args.num_tasks)
        ]

        for i in range(args.num_tasks):
            splits[i][0].targets = np.array(splits[i][0].targets) % num_cls
            splits[i][1].targets = np.array(splits[i][1].targets) % num_cls
            splits[i][2].targets = np.array(splits[i][2].targets) % num_cls

        [print(perm[num_cls * i:num_cls * (i+1)]) for i in range(args.num_tasks)]

        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        self.loaders = [
            (
                torch.utils.data.DataLoader(
                    x[0], batch_size=args.batch_size, shuffle=True, **kwargs
                ),
                torch.utils.data.DataLoader(
                    x[1], batch_size=args.test_batch_size, shuffle=True, **kwargs
                ),
                torch.utils.data.DataLoader(
                    x[2], batch_size=args.cal_batch_size, shuffle=True, **kwargs
                ),
            )
            for x in splits
        ]

    def update_task(self, i):
        self.train_loader = self.loaders[i][0]
        self.val_loader = self.loaders[i][1]
        self.cal_loader = self.loaders[i][2]

class RandSplitCIFAR100CSI:
    def __init__(self):
        super(RandSplitCIFAR100CSI, self).__init__()
        data_root = args.data
        num_cls = args.output_size

        use_cuda = torch.cuda.is_available()

        train_dataset = datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
        )
        val_dataset = datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(), 
                # normalize
                ]),
        )

        np.random.seed(args.seed)
        perm = np.arange(args.total_cls)
        print(perm)

        splits = [
            (
                partition_datasetv4_csi(train_dataset, perm[num_cls * i:num_cls * (i+1)], train=True, cal=False, val=False, prop=0.9) if args.validation \
                    else partition_datasetv4_csi(train_dataset, perm[num_cls * i:num_cls * (i+1)], train=True, cal=False),
                partition_datasetv4_csi(train_dataset, perm[num_cls * i:num_cls * (i+1)], train=True, cal=False, val=True, prop=0.9) if args.validation \
                    else partition_datasetv4_csi(val_dataset, perm[num_cls * i:num_cls * (i+1)], train=False, cal=False),
                partition_datasetv4_csi(train_dataset, perm[num_cls * i:num_cls * (i+1)], train=True, cal=True, val=False, prop=0.9) if args.validation \
                    else partition_datasetv4_csi(train_dataset, perm[num_cls * i:num_cls * (i+1)], train=True, cal=True),
            )
            for i in range(args.num_tasks)
        ]

        for i in range(args.num_tasks):
            splits[i][0].targets = np.array(splits[i][0].targets) % num_cls
            splits[i][1].targets = np.array(splits[i][1].targets) % num_cls
            splits[i][2].targets = np.array(splits[i][2].targets) % num_cls

        [print(perm[num_cls * i:num_cls * (i+1)]) for i in range(args.num_tasks)]

        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        self.loaders = [
            (
                torch.utils.data.DataLoader(
                    x[0], batch_size=args.batch_size, shuffle=True, **kwargs
                ),
                torch.utils.data.DataLoader(
                    x[1], batch_size=args.test_batch_size, shuffle=True, **kwargs
                ),
                torch.utils.data.DataLoader(
                    x[2], batch_size=args.cal_batch_size, shuffle=True, **kwargs
                ),
            )
            for x in splits
        ]

    def update_task(self, i):
        self.train_loader = self.loaders[i][0]
        self.val_loader = self.loaders[i][1]
        self.cal_loader = self.loaders[i][2]

class RandSplitTinyImgCSI:
    def __init__(self):
        super(RandSplitTinyImgCSI, self).__init__()
        # data_root = os.path.join(args.data, "cifar100")
        num_cls = args.output_size
        use_cuda = torch.cuda.is_available()

        train_dataset = ImageFolder(
            root=IMAGENET_PATH + '/train',
            transform=transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                ]
            ),
        )
        val_dataset = ImageFolder(
            root=IMAGENET_PATH + '/val_folders',
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                ]),
        )

        np.random.seed(args.seed)
        perm = np.arange(args.total_cls)
        print(perm)

        splits = [
            (
                Subset(train_dataset, partition_datasetv4_imgnet_csi(train_dataset, perm[num_cls * i:num_cls * (i+1)], train=True, cal=False, val=False, prop=0.9)) if args.validation \
                    else Subset(train_dataset, partition_datasetv4_imgnet_csi(train_dataset, perm[num_cls * i:num_cls * (i+1)], train=True, cal=False)),
                Subset(train_dataset, partition_datasetv4_imgnet_csi(train_dataset, perm[num_cls * i:num_cls * (i+1)], train=True, cal=False, val=True, prop=0.9)) if args.validation \
                    else Subset(val_dataset, partition_datasetv4_imgnet_csi(val_dataset, perm[num_cls * i:num_cls * (i+1)], train=False, cal=False)),
                Subset(train_dataset, partition_datasetv4_imgnet_csi(train_dataset, perm[num_cls * i:num_cls * (i+1)], train=True, cal=True, val=False, prop=0.9)) if args.validation \
                    else Subset(train_dataset, partition_datasetv4_imgnet_csi(train_dataset, perm[num_cls * i:num_cls * (i+1)], train=True, cal=True)),
            )
            for i in range(args.num_tasks)
        ]

        for i in range(args.num_tasks):
            splits[i][0].dataset.targets = np.array(splits[i][0].dataset.targets) % num_cls
            splits[i][1].dataset.targets = np.array(splits[i][1].dataset.targets) % num_cls
            splits[i][2].dataset.targets = np.array(splits[i][2].dataset.targets) % num_cls

        [print(perm[num_cls * i:num_cls * (i+1)]) for i in range(args.num_tasks)]

        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        self.loaders = [
            (
                torch.utils.data.DataLoader(
                    x[0], batch_size=args.batch_size, shuffle=True, **kwargs
                ),
                torch.utils.data.DataLoader(
                    x[1], batch_size=args.test_batch_size, shuffle=True, **kwargs
                ),
                torch.utils.data.DataLoader(
                    x[2], batch_size=args.cal_batch_size, shuffle=True, **kwargs
                ),
            )
            for x in splits
        ]
        # for i, x in enumerate(splits):
        #     print(i, x[0])

    def update_task(self, i):
        self.train_loader = self.loaders[i][0]
        self.val_loader = self.loaders[i][1]
        self.cal_loader = self.loaders[i][2]
