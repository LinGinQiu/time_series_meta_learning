import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sktime.datasets import load_from_tsfile

class FewShotDataset(Dataset):
    """
    FewShotDataset for time series classification (n, 1, length).
    Designed to load data from UCR datasets and generate N-way K-shot tasks.
    """
    def __init__(self, data_path, dataset_list_file, num_classes=5, num_samples_per_class=10):
        """
        :param data_path: Path to the UCR datasets.
        :param dataset_list_file: TXT file containing the list of dataset names to use.
        :param num_classes: Number of classes per task (N-way).
        :param num_samples_per_class: Number of samples per class (K-shot + Query).
        """
        self.data_path = data_path
        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class

        # Load dataset names from the TXT file
        with open(dataset_list_file, 'r') as f:
            self.datasets = [line.strip() for line in f.readlines()]

        # Load all data into memory
        self.data = {}
        for dataset in self.datasets:
            train_x, train_y = load_from_tsfile(os.path.join(data_path, dataset, dataset + "_TRAIN.ts"))
            test_x, test_y = load_from_tsfile(os.path.join(data_path, dataset, dataset + "_TEST.ts"))

            # Combine train and test data
            all_x = np.concatenate([train_x, test_x], axis=0)
            all_y = np.concatenate([train_y, test_y], axis=0)

            # Organize by class
            self.data[dataset] = {}
            for label in np.unique(all_y):
                self.data[dataset][label] = all_x