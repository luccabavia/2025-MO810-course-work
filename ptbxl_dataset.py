from api.MO810_API import MO810Dataset, MO810DataModule

from torch.utils.data import Dataset

import os
import glob

from PIL import Image

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Normalize, Resize, Lambda

import numpy as np
import pandas as pd
import wfdb
import ast

class PTBXLDataset(MO810Dataset):
    """
    Custom PyTorch Dataset for loading a PTB-XL ECG classification dataset.
    
    The dataset is organized by class folders. If download=True, the dataset is
    downloaded and unzipped from a remote Kaggle link. The dataset supports 
    optional transformations.
    """

    def __init__(
        self, 
        data_dir, 
        download=True, 
        transform=None, 
        sampling_rate: int = 500, 
        subset: str = None
    ):
        """
        Initializes the PTBXLDataset.
        Args:
            data_dir (str): Directory to store or load the dataset from.
            download (bool): Whether to download the dataset if not found.
            transform (callable, optional): Transformations to apply to each image.
        """
        super().__init__()
        # Data dir
        self.data_dir = data_dir
        self.dataset_dir = self.data_dir+"/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
        
        if download:
            self.remote_url = "https://www.kaggle.com/api/v1/datasets/download/khyeh0719/ptb-xl-dataset"
            self.local_filename = "ptb-xl-dataset.zip"
            self.download_data()

        self._samplign_rate: int = sampling_rate

        self.transform = transform   
        self._load_labels()

    def download_data(self):
        """
        Downloads and unzips the dataset if not already available locally.
        """
        if not os.path.exists(self.local_filename):
            # Download the dataset zip file 
            import urllib.request
            print(f"Downloading {self.local_filename} from {self.remote_url}")
            urllib.request.urlretrieve(self.remote_url, self.local_filename)

        if not os.path.exists(self.data_dir):
            print(f"Creatint the data folder")
            os.mkdir(self.data_dir)

        if not os.path.exists(self.dataset_dir):
            # Unzip the dataset file
            print(f"{self.dataset_dir}")
            import zipfile
            print(f"Extracting {self.local_filename} into {self.data_dir}")
            with zipfile.ZipFile(self.local_filename, 'r') as zip_ref:
                zip_ref.extractall(path=self.data_dir)

    def _load_labels(self) -> list:

        scp_description_databaset_path = self.dataset_dir + "/scp_statements.csv"
        scp_description_db = pd.read_csv(scp_description_databaset_path, sep=",", index_col=0)
        # scp_description_db = scp_description_db.dropna(subset=["diagnostic_class"])
        # available_classes = scp_description_db.diagnostic_class.unique()
        self._scp_description_db = scp_description_db

        database_path = self.dataset_dir + "/ptbxl_database.csv"
        database = pd.read_csv(database_path, index_col="ecg_id")
        database.scp_codes = database.scp_codes.apply(lambda x: ast.literal_eval(x))
        self._database = database

        classes = {code:np.zeros(shape=len(self._database.index), dtype=np.uint8) for code in self._scp_description_db.diagnostic_class.dropna().unique()}
        self.classes = list(classes.keys())

        for i, scp in enumerate(self._database.scp_codes):
        
            classes_keys = list()
            for key, val in scp.items():
                if val > 0:
                    entry = self._scp_description_db.loc[key]
                    classes_keys.append(entry.diagnostic_class)
            
            for key, val in classes.items():
                for k in classes_keys: 
                    if key == k:
                        val[i] = 1
        
        for key, val in classes.items():
            self._database[key] = val

        # Remove multilable and no label entries
        classes_per_line = list()
        multilabel = 0
        no_label = 0
        lines_to_maintain = list()
        
        sup_classes = self.classes
        for i in range(len(self._database)):
            c = list()
            for s in sup_classes:
                # print(s)
                # print(s, database[s].iloc[i])
                if self._database[s].iloc[i] == 1: c.append(s)
            if len(c) > 1: 
                # Droping multilabel lines
                multilabel += 1
                lines_to_maintain.append(0)
            elif len(c) == 0: 
                # Droping no label lines
                no_label += 1
                lines_to_maintain.append(0)
            else:
                lines_to_maintain.append(1)
                classes_per_line.append(c)  
        
        self._database = self._database[np.array(lines_to_maintain).astype(bool)]

        self.labels = list()
        for i in range(len(self._database)):
            for label_idx, s in enumerate(sup_classes):
                # print(s)
                # print(s, database[s].iloc[i])
                if self._database[s].iloc[i] == 1: 
                    self.labels.append(label_idx)
        self.labels = np.array(self.labels)

        self.nsamples = {}
        for c in self.classes:
            self.nsamples[c] = int(self._database[c].sum())
        
    def __len__(self):
        """
        Returns:
            int: Total number of samples in the dataset.
        """
        return len(self._database)

    def __getitem__(self, index):
        """
        Fetches a single image-label pair by index.
        Args:
            index (int): Index of the sample.
        Returns:
            tuple: (image, label), where image is a transformed tensor and label is an integer.
        """
        file_path = self._database.filename_hr.values[index]
        label = self.labels[index]
        
        data = wfdb.rdsamp(self.dataset_dir+"/"+file_path)
        image = data[0].T
        # image = image[:, None, :]

        if self.transform:
            image = self.transform(image)
        return image, label
    
    def __str__(self):
        """
        Returns:
            str: Human-readable description of the dataset.
        """
        return f"PTBXLDataset (# samples = {self.nsamples})"

class TransformedSubset(Dataset):
    """
    A dataset wrapper for applying a different transform to a subset of a dataset.

    This is useful when you want to change the transformation pipeline (e.g., data augmentation)
    for a specific subset, such as using a different transform for validation or testing
    while keeping the original dataset unchanged.

    Attributes:
        subset (torch.utils.data.Subset): The original subset of the dataset.
        transform (callable, optional): A function/transform that takes in a data sample and returns a transformed version.
    """

    def __init__(self, subset, transform=None):
        """
        Initialize the TransformedSubset.
        Args:
            subset (torch.utils.data.Subset): The subset to wrap.
            transform (callable, optional): Transform to apply to the input data.
        """
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        """
        Retrieve an item from the subset and apply the transform to the data.
        Args:
            idx (int): Index of the data sample to retrieve.
        Returns:
            tuple: (transformed_data, target), where target is the label or ground truth.
        """
        data, target = self.subset[idx]
        if self.transform:
            data = self.transform(data)
        return data, target

    def __len__(self):
        """
        Get the number of samples in the subset.
        Returns:
            int: Length of the dataset.
        """
        return len(self.subset)

class PTBXLDataModule(MO810DataModule):
    """
    PyTorch Lightning DataModule for the SneakersDataset.
    Handles data loading, splitting into train/val/test subsets,
    and setting up data loaders.
    """

    def __init__(self, data_dir: str = "./data/", 
                 train_transform = None,
                 val_transform = None,
                 test_transform = None,
                 batch_size: int = 32, 
                 num_workers: int = 4, 
                 use_recommended_splits: bool = True
                 ):
        """
        Initializes the SneakersDataModule.

        Default transform pipeline: ToImage() => reize((128,128)) => ToDtype(float32, scale=True) => Normalize())

        Args:
            data_dir (str): Directory where the dataset is stored or downloaded.
            train_transform: Transform pipeline to be applied to the train set. If None, the default pipeline is applied.
            val_transform: Transform pipeline to be applied to the validation set. If None, the default pipeline is applied.
            test_transform: Transform pipeline to be applied to the test set. If None, the default pipeline is applied.
            batch_size (int): Batch size for the data loaders.
            num_workers (int): Number of subprocesses for data loading.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_recommended_splits = use_recommended_splits

        # Precomputed normalization stats (post-transforms)
        # Transforms: ToImage(), Resize((128,128)), and ToDtype(scale=True)
        self.precomputed_dataset_stats = {
            'mean': torch.tensor([-0.0019, -0.0015,  0.0005,  0.0017, -0.0011, -0.0005,  0.0003, -0.0010,-0.0017, -0.0021, -0.0012, -0.0011]),
            'std': torch.tensor([0.1699, 0.1573, 0.1645, 0.1416, 0.1476, 0.1366, 0.2178, 0.3177, 0.3082, 0.2941, 0.2830, 0.2346])}

        # Validation and test set image transformation pipeline
        self.default_transform_pipeline = Compose([Lambda(lambda x: torch.from_numpy(x)), 
                                                   ToDtype(torch.float32, scale=True),
                                                   Lambda(lambda x: (x - self.precomputed_dataset_stats["mean"].unsqueeze(1)) / self.precomputed_dataset_stats["std"].unsqueeze(1))]
                                                   )

        # Load full dataset with transforms
        self.full_dataset = PTBXLDataset(data_dir=self.data_dir)

        # Split into training, validation, and test subsets
        self.train_subset, self.val_subset, self.test_subset = self.split(self.full_dataset)

        # Set the transform pipelines
        if train_transform:
            self.train_transform = train_transform
        else:
            self.train_transform = self.default_transform_pipeline
        if val_transform:
            self.val_transform = val_transform
        else:
            self.val_transform = self.default_transform_pipeline
        if test_transform:
            self.test_transform = test_transform
        else:
            self.test_transform = self.default_transform_pipeline

    def split(self, dataset):
        """
        Splits the dataset into train, val, and test sets.

        Args:
            dataset (Dataset): The full dataset to split.

        Returns:
            tuple: (train_subset, val_subset, test_subset)
        """

        if self.use_recommended_splits:
            # Use splits recommended by dataset authors
            train_subset = Subset(dataset, np.ravel(np.argwhere(dataset._database.strat_fold < 9)))
            val_subset = Subset(dataset, np.ravel(np.where(dataset._database.strat_fold == 9)))
            test_subset = Subset(dataset, np.ravel(np.where(dataset._database.strat_fold == 10)))

            return train_subset, val_subset, test_subset
        else:
            # 20% for test
            test_size = int(0.2 * len(dataset))
            # 64% for train (80% of non-test samples)
            train_size = int(0.8 * (len(dataset) - test_size))
            # 16% for validation (20% of non-test samples)
            val_size = len(dataset) - train_size - test_size

            generator = torch.Generator().manual_seed(42)
            return random_split(dataset, [train_size, val_size, test_size], generator=generator)

    def sample_dataset(self, dataset, fraction=None, samples_per_class=None):
        """
        Returns a sampled subset of the dataset based on a fraction of the total dataset
        or a fixed number of samples per class.
        Args:
            dataset (Dataset): The dataset to sample from.
            fraction (float, optional): Fraction of the dataset to use (0 < fraction ≤ 1).
            samples_per_class (int, optional): Number of samples per class to include.

        Returns:
            Subset: A subset of the dataset with the selected samples.

        Raises:
            ValueError: If both `fraction` and `samples_per_class` are provided.
        """
        if fraction != None and samples_per_class != None:
            raise ValueError("SneakersDataModule ERROR: Cannot sample dataset using both fraction and samples_per_class")
        elif fraction:
            N = int(len(dataset) * fraction)
            return Subset(dataset, range(N))
        elif samples_per_class:
            count = [0] * len(self.full_dataset.classes)
            indices = []
            for i, (_, label) in enumerate(dataset):
                if count[label] < samples_per_class: 
                    count[label] += 1
                    indices.append(i)
            return Subset(dataset, indices)
        else:
            return dataset

    def train_dataloader(self, fraction=None, samples_per_class=None, transform=None):
        """
        Returns a dataloader for the training set.
        Args:
            fraction (float, optional): Fraction of the training dataset to use (0 < fraction ≤ 1).
            samples_per_class (int, optional): Number of samples per class to include.
            transform: If != None override the transform defined at init.

        Returns:
            DataLoader: DataLoader for training set.
        """
        if not transform: transform = self.train_transform
        train_subset = self.sample_dataset(self.train_subset, fraction, samples_per_class)
        return DataLoader(dataset=TransformedSubset(subset=train_subset, 
                                                    transform=transform),
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers)

    def val_dataloader(self, transform=None):
        """
        Args:
            transform: If != None override the transform defined at init.
        Returns:
            DataLoader: DataLoader for validation set.
        """
        if not transform: transform = self.val_transform
        return DataLoader(dataset=TransformedSubset(subset=self.val_subset, 
                                                    transform=transform),
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers)

    def test_dataloader(self, transform=None):
        """
        Args:
            transform: If != None override the transform defined at init.
        Returns:
            DataLoader: DataLoader for test set.
        """
        if not transform: transform = self.test_transform
        return DataLoader(dataset=TransformedSubset(subset=self.test_subset, 
                                                    transform=transform),
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers)
    