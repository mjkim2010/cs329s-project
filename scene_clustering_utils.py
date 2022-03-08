#!/usr/bin/env python3
import os
import pickle
if 'x86' in os.uname().machine:
    from sklearnex import patch_sklearn
    patch_sklearn()
from cache_utils import SimpleLRUCache, get_img_hash
import torch
import torchvision.models as models
from torchvision import transforms as trn
import torch.nn as nn
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN
import numpy as np

CACHE_MAX_SIZE = 10000
EMBED_DIM = 2048
PLACES365_BATCH_SIZE = 128
AVAILABLE_PRETRAINED_KMEANS_MODELS = {'inside_outside'}

class BasicPlacesDataset(Dataset):
    def __init__(self, imgs=None, img_fps=None):
        if imgs is not None:
            self.imgs = imgs
        elif img_fps is not None:
            self.img_fps = img_fps
        else:
            raise ValueError('Either imgs or img_fps must be non-null')
        self.centre_crop = trn.Compose([
            trn.ToTensor(),
            trn.Resize((256,256)),
            trn.CenterCrop(224),
        ])
        self.norm = trn.Compose([
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        if hasattr(self, 'imgs'):
            img = self.imgs[idx]
        else:
            img = Image.open(self.img_fps[idx])
        input_img = self.centre_crop(img)
        if input_img.shape[0] != 3: # handle greyscale images
            input_img = torch.cat((input_img, input_img, input_img), dim=0)
        assert(input_img.shape == (3, 224, 224))
        input_img = self.norm(input_img)
        return input_img

    def __len__(self):
        return len(self.imgs) if hasattr(self, 'imgs') else len(self.img_fps)

def load_pretrained_model(arch: str, remove_last_layer=True):
    # load the pre-trained weights
    model_path = arch + '_places365.pth.tar'
    if not os.access(model_path, os.W_OK):
        print(f'Downloading pretrained {arch} to current directory...')
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_path
        os.system('wget ' + weight_url)

    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False
    if remove_last_layer: # to extract embeds
        modules = list(model.children())[:-1]
        model = nn.Sequential(*modules)
    return model

class ImageClusterer:
    def __init__(self, arch='resnet50', use_cache=True):
        self.model = load_pretrained_model(arch, remove_last_layer=True)
        self.use_cache = use_cache
        if self.use_cache:
            self.embeds_cache = SimpleLRUCache(CACHE_MAX_SIZE)

    def update_cached_embeds(self, img_fps, embeds):
        noncached_idxs, noncached_hashes = [], []
        img_hashes = [get_img_hash(fp) for fp in img_fps]
        for idx, img_hash in enumerate(img_hashes):
            if img_hash in self.embeds_cache:
                embeds[idx] = self.embeds_cache[img_hash]
            else:
                noncached_idxs.append(idx)
                noncached_hashes.append(img_hash)
        return noncached_idxs, noncached_hashes, embeds

    def extract_embeds(self, img_fps):
        """
        Basic problem: I want to get cached embeds for some fps,
        and extract embeds as normal for other embeds.
        """
        embeds = torch.zeros(len(img_fps), EMBED_DIM)
        if self.use_cache:
            noncached_idxs, noncached_hashes, embeds = self.update_cached_embeds(img_fps, embeds)
            num_cached = len(img_fps) - len(noncached_idxs)
            print(f'Number of cached embeds: {num_cached} ({100 * num_cached / len(img_fps)}%)')
            if not noncached_idxs: # Everything in cache; return
                return embeds
            noncached_img_fps = np.array(img_fps)[noncached_idxs]
        else:
            noncached_img_fps = img_fps

        dataset = BasicPlacesDataset(img_fps=noncached_img_fps)
        data_loader = DataLoader(dataset, batch_size=PLACES365_BATCH_SIZE, shuffle=False)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using device \'{device}\' to extract embeds...')
        self.model = self.model.to(device)
        noncached_embeds = []
        for imgs in tqdm(data_loader):
            imgs = imgs.to(device)
            out_embeds = self.model(imgs)
            out_embeds = out_embeds.view(*out_embeds.shape[:2])
            noncached_embeds.append(out_embeds)
        noncached_embeds = torch.cat(noncached_embeds, dim=0)
        noncached_embeds = noncached_embeds.cpu()
        if self.use_cache:
            for i, (idx, img_hash) in enumerate(zip(noncached_idxs, noncached_hashes)):
                self.embeds_cache[img_hash] = noncached_embeds[i]
                embeds[idx] = noncached_embeds[i]
        else:
            embeds = noncached_embeds
        return embeds

    def cluster_kmeans(self, img_fps, n_clusters=None, pretrained_kmeans=None):
        if n_clusters is None and pretrained_kmeans is None:
            raise ValueError('Either n_clusters or pretrained_kmeans must be non-null')
        embeds = self.extract_embeds(img_fps)
        if pretrained_kmeans is not None:
            if pretrained_kmeans not in AVAILABLE_PRETRAINED_KMEANS_MODELS:
                raise ValueError('Unexpected value for pretrained_kmeans')
            with open(f'{pretrained_kmeans}_kmeans.p', 'rb') as f:
                kmeans = pickle.load(f)
            clusters = kmeans.predict(embeds)
        else:
            clusters = KMeans(n_clusters, random_state=31415926).fit_predict(embeds)
        return clusters

    def cluster_dbscan(self, img_fps, eps=11, min_samples=5):
        embeds = self.extract_embeds(img_fps)
        clusters = DBSCAN(eps, min_samples=min_samples).fit_predict(embeds)
        return clusters
