import torch


class CachedDataset(torch.utils.data.Dataset):
    """
    Wrapper for a dataset that exhaustively caches the return values of __getitem__ in the constructor.

    Note that this wrapper is not applicable to datasets that do data augmentation (MLP, ACT, DiffusionPolicy) or randomly retrieve indexes (ACT) in __getitem__.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        print(
            f"[{self.__class__.__name__}] Start cache creation of {self.dataset.__class__.__name__}."
        )
        self.data_list = [self.dataset[i] for i in range(len(self.dataset))]
        print(
            f"[{self.__class__.__name__}] Finished cache creation of {self.dataset.__class__.__name__}."
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

    @property
    def filenames(self):
        return self.dataset.filenames

    @property
    def image_transforms(self):
        return self.dataset.image_transforms
