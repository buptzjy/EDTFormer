# https://github.com/amaralibey/gsv-cities

import pandas as pd
import random
import numpy as np
import logging
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# NOTE: Hard coded path to dataset folder 
BASE_PATH = '/data_nvme/zhangjingyi/Gsv_reflect/mixgsv/'

if not Path(BASE_PATH).exists():
    raise FileNotFoundError(
        'BASE_PATH is hardcoded, please adjust to point to gsv_reflect/mixgsv')

def is_synthetic_panoid(panoid):
    """Check if a panoid corresponds to a synthetic image.
    Synthetic images have '__reflectvpr_' in their panoid."""
    return '__reflectvpr_' in str(panoid)


def normalize_training_subsets(training_subsets):
    normalized = []
    for subset in training_subsets or ["default"]:
        subset_name = str(subset).strip()
        if subset_name:
            normalized.append(subset_name)
    return normalized or ["default"]


class GSVCitiesDataset(Dataset):
    def __init__(self,
                 cities=['London', 'Boston'],
                 img_per_place=4,
                 min_img_per_place=4,
                 random_sample_from_each_place=True,
                 transform=default_transform,
                 base_path=BASE_PATH,
                 synthetic_ratio=0.0,
                 training_subsets=None,
                 tmp_group="all",
                 ):
        super(GSVCitiesDataset, self).__init__()
        self.base_path = Path(base_path).expanduser()
        self.cities = cities

        assert img_per_place <= min_img_per_place, \
            f"img_per_place should be less than {min_img_per_place}"
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform
        self.synthetic_ratio = synthetic_ratio
        self.training_subsets = normalize_training_subsets(training_subsets)
        self.tmp_group = tmp_group

        # generate the dataframe containing images metadata
        self.dataframe = self.__getdataframes()

        # get all unique place ids
        self.places_ids = pd.unique(self.dataframe.index)
        self.total_nb_images = len(self.dataframe)

    @staticmethod
    def is_tmp_pitts_csv(csv_path):
        stem = csv_path.stem.strip().lower()
        if stem.startswith("pitts30k"):
            suffix = stem[len("pitts30k"):]
        elif stem.startswith("pitts"):
            suffix = stem[len("pitts"):]
        else:
            return False
        return suffix.isdigit() and 0 <= int(suffix) <= 17

    @classmethod
    def should_include_csv(cls, subset, csv_path, tmp_group):
        if subset != "tmp" or tmp_group == "all":
            return True
        is_pitts_csv = cls.is_tmp_pitts_csv(csv_path)
        if tmp_group == "pitts":
            return is_pitts_csv
        if tmp_group == "msls":
            return not is_pitts_csv
        return True

    def collect_csv_sources(self):
        dataframes_dir = self.base_path / "Dataframes"
        images_root = self.base_path / "Images"
        csv_sources = []
        seen_keys = set()

        for subset in self.training_subsets:
            subset_dir = dataframes_dir if subset == "default" else dataframes_dir / subset
            if not subset_dir.is_dir():
                logging.warning(f"Training subset directory not found, skipping: {subset_dir}")
                continue

            subset_images_dir = images_root if subset == "default" else images_root / subset
            active_images_dir = subset_images_dir if subset_images_dir.is_dir() else images_root

            for csv_path in sorted(subset_dir.glob("*.csv")):
                if not self.should_include_csv(subset, csv_path, self.tmp_group):
                    continue
                city_name = csv_path.stem
                if self.cities != "all" and city_name not in self.cities:
                    continue
                source_key = (subset, city_name)
                if source_key in seen_keys:
                    continue
                seen_keys.add(source_key)
                csv_sources.append({
                    "city_name": city_name if subset == "default" else f"{subset}/{city_name}",
                    "csv_path": csv_path,
                    "images_dir": active_images_dir,
                })

        return csv_sources

    def __getdataframes(self):
        ''' 
            Return one dataframe containing
            all info about the images from all cities

            This requieres DataFrame files to be in a folder
            named Dataframes, containing a DataFrame
            for each city in self.cities
        '''
        csv_sources = self.collect_csv_sources()
        if not csv_sources:
            raise FileNotFoundError(
                f"No CSV files found in {self.base_path / 'Dataframes'} "
                f"for subsets {self.training_subsets} and tmp_group={self.tmp_group}"
            )

        dataframes = []
        self.cities = []
        for i, source in enumerate(csv_sources):
            tmp_df = pd.read_csv(source["csv_path"])
            tmp_df["__images_dir"] = str(source["images_dir"])
            self.cities.append(source["city_name"])

            # Now we add a prefix to place_id, so that we
            # don't confuse, say, place number 13 of NewYork
            # with place number 13 of London ==> (0000013 and 0500013)
            # We suppose that there is no city with more than
            # 99999 images and there won't be more than 99 cities
            # TODO: rename the dataset and hardcode these prefixes
            prefix = i
            tmp_df['place_id'] = tmp_df['place_id'] + (prefix * 10**5)
            tmp_df = tmp_df.sample(frac=1)  # shuffle the city dataframe
            dataframes.append(tmp_df)

        df = pd.concat(dataframes, ignore_index=True)

        # keep only places depicted by at least min_img_per_place images
        res = df[df.groupby('place_id')['place_id'].transform(
            'size') >= self.min_img_per_place]
        return res.set_index('place_id')

    def _sample_images_with_ratio(self, place_df):
        """Sample img_per_place images from a place, respecting synthetic_ratio.

        For each place, we sample:
        - synthetic images: according to synthetic_ratio (with random rounding)
        - real images: fill the rest

        Random rounding:
            expected_syn = img_per_place * synthetic_ratio
            syn_num = floor(expected_syn)
            if random() < expected_syn - floor(expected_syn):
                syn_num += 1

        If not enough synthetic/real images, fill with the other type.
        """
        real_df = place_df[~place_df['panoid'].apply(is_synthetic_panoid)]
        syn_df = place_df[place_df['panoid'].apply(is_synthetic_panoid)]

        if self.synthetic_ratio <= 0 or len(syn_df) == 0:
            # Real-only mode
            sampled = real_df.sample(n=min(self.img_per_place, len(real_df)),
                                     replace=False) if len(real_df) > 0 else real_df
            # Pad with synthetic if not enough real
            if len(sampled) < self.img_per_place and len(syn_df) > 0:
                extra = syn_df.sample(n=self.img_per_place - len(sampled), replace=True)
                sampled = pd.concat([sampled, extra])
            return sampled

        # Random rounding to determine synthetic count
        expected_syn = self.img_per_place * self.synthetic_ratio
        syn_num = int(np.floor(expected_syn))
        if random.random() < expected_syn - np.floor(expected_syn):
            syn_num += 1

        # Clamp to available images
        syn_num = min(syn_num, self.img_per_place)
        real_num = self.img_per_place - syn_num

        # Sample synthetic images
        syn_sampled = syn_df.sample(n=min(syn_num, len(syn_df)),
                                    replace=False) if len(syn_df) > 0 else syn_df
        # Sample real images
        real_sampled = real_df.sample(n=min(real_num, len(real_df)),
                                      replace=False) if len(real_df) > 0 else real_df

        # If not enough synthetic, fill with real
        if len(syn_sampled) < syn_num:
            extra_real_needed = syn_num - len(syn_sampled)
            extra_real = real_df.drop(real_sampled.index, errors='ignore')
            extra_real = extra_real.sample(n=min(extra_real_needed, len(extra_real)),
                                           replace=True) if len(extra_real) > 0 else extra_real
            real_sampled = pd.concat([real_sampled, extra_real])

        # If not enough real, fill with synthetic
        if len(real_sampled) < real_num:
            extra_syn_needed = real_num - len(real_sampled)
            extra_syn = syn_df.drop(syn_sampled.index, errors='ignore')
            extra_syn = extra_syn.sample(n=min(extra_syn_needed, len(extra_syn)),
                                         replace=True) if len(extra_syn) > 0 else extra_syn
            syn_sampled = pd.concat([syn_sampled, extra_syn])

        # Combine and sample exactly img_per_place
        sampled = pd.concat([real_sampled, syn_sampled])
        if len(sampled) > self.img_per_place:
            sampled = sampled.sample(n=self.img_per_place)
        elif len(sampled) < self.img_per_place:
            sampled = sampled.sample(n=self.img_per_place, replace=True)

        return sampled

    def __getitem__(self, index):
        place_id = self.places_ids[index]

        # get the place in form of a dataframe (each row corresponds to one image)
        place = self.dataframe.loc[place_id]

        # Ensure place is a DataFrame (handle single-row case)
        if isinstance(place, pd.Series):
            place = place.to_frame().T

        if self.random_sample_from_each_place:
            if self.synthetic_ratio > 0:
                place = self._sample_images_with_ratio(place)
            else:
                place = place.sample(n=self.img_per_place)
        else:
            place = place.sort_values(
                by=['year', 'month', 'lat'], ascending=False)
            place = place[: self.img_per_place]

        imgs = []
        remaining = list(place.iterrows())
        while len(imgs) < self.img_per_place and remaining:
            idx, row = remaining.pop(0)
            img_name = self.get_img_name(row)
            img_path = Path(row['__images_dir']) / row['city_id'] / img_name
            try:
                img = self.image_loader(img_path)
            except (FileNotFoundError, OSError):
                continue

            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)

        # Pad with the last image if not enough images were loaded
        while len(imgs) < self.img_per_place:
            imgs.append(imgs[-1] if imgs else torch.zeros(3, 224, 224))

        # NOTE: contrary to image classification where __getitem__ returns only one image 
        # in GSVCities, we return a place, which is a Tesor of K images (K=self.img_per_place)
        # this will return a Tensor of shape [K, channels, height, width]. This needs to be taken into account 
        # in the Dataloader (which will yield batches of shape [BS, K, channels, height, width])
        return torch.stack(imgs), torch.tensor(place_id).repeat(self.img_per_place)

    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return len(self.places_ids)

    @staticmethod
    def image_loader(path):
        return Image.open(path).convert('RGB')

    @staticmethod
    def get_img_name(row):
        # given a row from the dataframe
        # return the corresponding image name

        city = row['city_id']

        # now remove the two digit we added to the id
        # they are superficially added to make ids different
        # for different cities
        pl_id = row.name % 10**5  #row.name is the index of the row, not to be confused with image name
        pl_id = str(pl_id).zfill(7)

        panoid = row['panoid']
        year = str(row['year']).zfill(4)
        month = str(row['month']).zfill(2)
        northdeg = str(row['northdeg']).zfill(3)
        lat, lon = str(row['lat']), str(row['lon'])
        name = city+'_'+pl_id+'_'+year+'_'+month+'_' + \
            northdeg+'_'+lat+'_'+lon+'_'+panoid+'.jpg'
        return name
