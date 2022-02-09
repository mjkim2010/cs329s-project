#!/usr/bin/env python3
import torch
import torchvision.models as models
from torchvision import transforms as trn
import torch.nn as nn
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.cluster import KMeans


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
    img = self.imgs[idx] if hasattr(self, 'imgs') else Image.open(self.img_fps[idx])
    input_img = self.centre_crop(img)
    if input_img.shape[0] != 3: # handle greyscale images
      input_img = torch.cat((input_img, input_img, input_img), dim=0)
    assert(input_img.shape == (3, 224, 224))
    input_img = self.norm(input_img)
    return input_img

  def __len__(self):
    return len(self.imgs)

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


def extract_embeds(model_truncated, imgs, embeds_output_path='img_scene_embeds.pt'):
    data_loader = DataLoader(BasicPlacesDataset(imgs), batch_size=128, shuffle=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device \'{device}\' to extract embeds...')
    model_truncated = model_truncated.to(device)
    embeds = []
    for imgs in tqdm(data_loader):
        imgs = imgs.to(device)
        out_embeds = model_truncated(imgs)
        out_embeds = out_embeds.view(*out_embeds.shape[:2])
        embeds.append(out_embeds)
    embeds = torch.cat(embeds, dim=0)
    embeds = embeds.cpu()
    torch.save(embeds, embeds_output_path)
    return embeds

class ImageClusterer:
    def __init__(self, arch='resnet50'):
        self.model = load_pretrained_model(arch, remove_last_layer=True)
    def __call__(self, imgs, n_clusters):
        embeds = extract_embeds(self.model, imgs)
        print(embeds.shape)
        clusters = KMeans(n_clusters, random_state=31415926, verbose=1).fit_predict(embeds)
        return clusters
