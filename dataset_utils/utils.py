import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split

###################################################

class Split_Dataset(Dataset):
    """
        Split a torch dataset with the same transform.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)


def split_dataset(dataset,percent:float) -> torch.utils.data.Dataset:
    """ 
        dataset : the dataset to split
        percent : float between 0 and 1. 
        Function use for split a dataset and use only a certain part for the supervise training.
    """
    torch.manual_seed(0)
    split = int(len(dataset)*percent)
    lengths = [split,len(dataset)-split]
    labeled, _ = random_split(dataset, lengths)
    train_full_supervised = Split_Dataset(
        labeled)
    torch.manual_seed(torch.initial_seed())
    return train_full_supervised


def my_collate(batch):
    print(len(batch))
    print(batch[0][0].size(),batch[0][1].size())
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return (data, target)