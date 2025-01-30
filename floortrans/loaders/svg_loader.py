import lmdb
import pickle
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from numpy import genfromtxt
from floortrans.loaders.house import House
import os


class FloorplanSVG(Dataset):
    def __init__(self, data_folder, data_file, format='txt', augmentations=None,
                 image_norm=True, original_size=False, lmdb_folder='cubi_lmdb/'):
        self.data_folder = data_folder
        self.data_file = data_file
        self.format = format
        self.augmentations = augmentations
        self.image_norm = image_norm
        self.original_size = original_size
        self.lmdb_folder = lmdb_folder

        print("Initializing FloorplanSVG dataset...")
        print("Data folder:", data_folder)
        print("Data file:", data_file)
        print("Is transform:", augmentations is not None)
        print("Augmentations:", augmentations)
        print("Image normalization:", image_norm)
        print("Format:", format)
        print("Original size:", original_size)
        print("LMDB folder:", lmdb_folder)

        # Initialize folders list as a proper list
        self.folders = []
        
        # Read folders from data file
        with open(os.path.join(data_folder, data_file), 'r') as f:
            for line in f:
                folder = line.strip()
                if folder:  # Skip empty lines
                    print("Folders:", folder)
                    self.folders.append(folder)

        self.img_norm = image_norm
        self.is_transform = augmentations is not None
        self.get_data = None
        self.image_file_name = '/F1_scaled.png'
        self.org_image_file_name = '/F1_original.png'
        self.svg_file_name = '/model.svg'

        if format == 'txt':
            self.get_data = self.get_txt
        if format == 'lmdb':
            self.lmdb = lmdb.open(data_folder+lmdb_folder, readonly=True,
                                  max_readers=8, lock=False,
                                  readahead=True, meminit=False)
            self.get_data = self.get_lmdb
            self.is_transform = False

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        print("Index:", index)
        sample = self.get_data(index)

        if self.augmentations is not None:
            sample = self.augmentations(sample)
            
        if self.is_transform:
            sample = self.transform(sample)

        return sample

    def get_txt(self, index):
        print("Getting data from txt file...")
        print("Index:", index)
        print("Folders:", self.folders)
        print("Data folder:", self.data_folder)
        print("Image file name:", self.image_file_name)
        print("SVG file name:", self.svg_file_name)
        fplan = cv2.imread(self.data_folder + self.folders[index] + self.image_file_name)
        fplan = cv2.cvtColor(fplan, cv2.COLOR_BGR2RGB)  # correct color channels
        height, width, nchannel = fplan.shape
        fplan = np.moveaxis(fplan, -1, 0)

        # Getting labels for segmentation and heatmaps
        house = House(self.data_folder + self.folders[index] + self.svg_file_name, height, width)
        # Combining them to one numpy tensor
        label = torch.tensor(house.get_segmentation_tensor().astype(np.float32))
        heatmaps = house.get_heatmap_dict()
        coef_width = 1
        if self.original_size:
            fplan = cv2.imread(self.data_folder + self.folders[index] + self.org_image_file_name)
            fplan = cv2.cvtColor(fplan, cv2.COLOR_BGR2RGB)  # correct color channels
            height_org, width_org, nchannel = fplan.shape
            fplan = np.moveaxis(fplan, -1, 0)
            label = label.unsqueeze(0)
            label = torch.nn.functional.interpolate(label,
                                                    size=(height_org, width_org),
                                                    mode='nearest')
            label = label.squeeze(0)

            coef_height = float(height_org) / float(height)
            coef_width = float(width_org) / float(width)
            for key, value in heatmaps.items():
                heatmaps[key] = [(int(round(x*coef_width)), int(round(y*coef_height))) for x, y in value]

        img = torch.tensor(fplan.astype(np.float32))

        sample = {'image': img, 'label': label, 'folder': self.folders[index],
                  'heatmaps': heatmaps, 'scale': coef_width}

        return sample

    def get_lmdb(self, index):
        key = self.folders[index].encode()
        with self.lmdb.begin(write=False) as f:
            data = f.get(key)

        sample = pickle.loads(data)
        return sample

    def transform(self, sample):
        fplan = sample['image']
        # Normalization values to range -1 and 1
        fplan = 2 * (fplan / 255.0) - 1

        sample['image'] = fplan

        return sample
