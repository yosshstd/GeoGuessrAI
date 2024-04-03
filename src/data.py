from PIL import Image
from tqdm import tqdm
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomResizedCrop
from transformers import CLIPImageProcessor, AutoTokenizer, CLIPTextModelWithProjection

import reverse_geocoder as rg
import pycountry



class CustomDataset(Dataset):
    def __init__(self, dataset_dir, model_name, augment):
        self.augment = augment
        self.paths = []
        self.coords = []

        for filepath in tqdm(glob.iglob(f'{dataset_dir}/*.jpg')):
            self.paths.append(filepath)
            coord_list = filepath.split("/")[-1].split(",")
            lat, lon = [float(coord_list[0]), float(coord_list[1][:-4])]
            self.coords.append((lat, lon))
        
        self.rg_results = rg.search(self.coords)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModelWithProjection.from_pretrained(model_name)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.augment:
            transform = RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(3/4, 4/3))
            image = transform(image)
        if self.processor is not None:
            image = self.processor(image, return_tensors="pt")['pixel_values'].squeeze(0)
        coord = self.coords[idx]
        country_cc = self.rg_results[idx]['cc']
        country_name = get_country_name(country_cc)
        admin_name = self.rg_results[idx]['admin1']
        name = self.rg_results[idx]['name']
        country_text = f"A street view photo in {name}, {admin_name}, {country_name}."
        tokenized_text = self.tokenizer(country_text, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_country_embedding = self.text_model(**tokenized_text).text_embeds.squeeze()
        return image, torch.tensor(coord), text_country_embedding, country_text

def get_dataloaders(dataset_dir, batch_size, model_name, augment):

    dataset = CustomDataset(dataset_dir, model_name, augment)
    SIZE = len(dataset)
    indices = torch.randperm(SIZE)
    split = int(SIZE * 0.9)
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Dataset size: {SIZE}")

    return train_loader, val_loader

def get_country_name(country_code):
    try:
        country = pycountry.countries.get(alpha_2=country_code)
        return country.name
    except AttributeError:
        return "Unknown"