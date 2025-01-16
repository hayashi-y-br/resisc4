import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


class Patchify(object):
        def __init__(self, patch_size=32):
            self.patch_size = patch_size

        def __call__(self, img):
            c = img.shape[0]
            img = img.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
            img = img.permute(1, 2, 0, 3, 4)
            img = img.contiguous().view(-1, c, self.patch_size, self.patch_size)
            return img


class RESISC4(Dataset):
    def __init__(self, root='./data', split='train', patch_size=32):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            Patchify(patch_size)
        ])
        dataset = datasets.ImageFolder(root=root, transform=transform)
        self.class_to_idx = dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        labels = np.array([label for _, label in dataset])
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=1/7, random_state=42)
        train_valid_indices, test_indices = next(sss_test.split(np.zeros(len(labels)), labels))
        train_valid_labels = labels[train_valid_indices]
        sss_valid = StratifiedShuffleSplit(n_splits=1, test_size=1/6, random_state=42)
        train_indices, valid_indices = next(sss_valid.split(np.zeros(len(train_valid_labels)), train_valid_labels))
        train_indices = train_valid_indices[train_indices]
        valid_indices = train_valid_indices[valid_indices]
        if split == 'train':
            self.dataset = Subset(dataset, train_indices)
        elif split == 'valid':
            self.dataset = Subset(dataset, valid_indices)
        elif split == 'test':
            self.dataset = Subset(dataset, test_indices)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)


if __name__ == '__main__':
    for split in ['train', 'valid', 'test']:
        dataset = RESISC4(split=split)
        print(f'{split}\t: {len(dataset):4d}')
        counts = np.zeros(4, dtype=int)
        for _, y in dataset:
            counts[y] += 1
        for i in range(4):
            print(f'class {i}\t: {counts[i]:4d}')