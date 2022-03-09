import torch
from torch import nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from cache_utils import SimpleLRUCache, get_img_hash
from skimage import io
import numpy as np
from tqdm import tqdm

CACHE_MAX_SIZE = 10000
BATCH_SIZE = 1


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class IQAModel(nn.Module):
    def __init__(self):
        super(IQAModel, self).__init__()
        transfered_model = torchvision.models.vgg16(pretrained=True)
        modules = list(transfered_model.children())[:-1]
        modules.append(nn.AvgPool2d(7, 1))
        flatten = Flatten()
        modules.append(flatten)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        self.fe = nn.Sequential(*modules)
        self.relu_ac = nn.ReLU()
        self.linear = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fe(x)
        x = self.linear(x)
        x = self.relu_ac(x)
        x = self.linear2(x)
        x = self.relu_ac(x)
        x = self.linear3(x)
        x = self.relu_ac(x)
        x = self.linear4(x)
        x = self.relu_ac(x)
        x = self.linear5(x)
        return x


class ImageDataset2(Dataset):
    def __init__(self, image_fps, device):
        self.image_fps = image_fps
        self.resize = transforms.Compose([
            transforms.Resize((768,1024)),
        ])
        self.norm = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.device = device

    def __getitem__(self, idx):
        input_img = transforms.ToTensor()(io.imread(self.image_fps[idx])).to(self.device)
        input_img = self.resize(input_img)
        if input_img.shape[0] != 3:  # handle greyscale images
            input_img = torch.cat((input_img, input_img, input_img), dim=0)
        X_b = self.norm(input_img)
        # X_b is image, X is file path, Y is label
        return X_b

    def __len__(self):
        return len(self.image_fps)


def load_model(state_dict_fp, device):
    model = IQAModel()
    model.to(device)
    model.load_state_dict(torch.load(state_dict_fp, map_location=device))
    model.eval()
    return model


class IQAClass:
    def __init__(self, state_dict_fp, use_cache=True):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model = load_model(state_dict_fp, self.device)
        self.use_cache = use_cache
        if self.use_cache:
            self.embeds_cache = SimpleLRUCache(CACHE_MAX_SIZE)

    def update_cached_embeds(self, img_fps, ratings):
        noncached_idxs, noncached_hashes = [], []
        img_hashes = [get_img_hash(fp) for fp in img_fps]
        for idx, img_hash in enumerate(img_hashes):
            if img_hash in self.embeds_cache:
                ratings[idx] = self.embeds_cache[img_hash]
            else:
                noncached_idxs.append(idx)
                noncached_hashes.append(img_hash)
        return noncached_idxs, noncached_hashes, ratings

    def extract_embeds(self, img_fps):
        """
        Basic problem: I want to get cached embeds for some fps,
        and extract embeds as normal for other embeds.
        """
        ratings = torch.zeros(len(img_fps))
        if self.use_cache:
            noncached_idxs, noncached_hashes, embeds = self.update_cached_embeds(img_fps, ratings)
            num_cached = len(img_fps) - len(noncached_idxs)
            print(f'Number of cached embeds: {num_cached} ({100 * num_cached / len(img_fps)}%)')
            if not noncached_idxs:  # Everything in cache; return
                return embeds
            noncached_img_fps = np.array(img_fps)[noncached_idxs]
        else:
            noncached_img_fps = img_fps
            noncached_idxs, noncached_hashes = None, None

        dataset = ImageDataset2(noncached_img_fps, self.device)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        print(f'Using device \'{self.device}\' to extract embeds...')
        noncached_scores = []
        for imgs in tqdm(data_loader):
            with torch.no_grad():
              scores = self.model.forward(imgs)
            noncached_scores.append(scores)
        noncached_scores = torch.cat(noncached_scores, dim=0)
        noncached_scores = noncached_scores.cpu()
        if self.use_cache:
            for i, (idx, img_hash) in enumerate(zip(noncached_idxs, noncached_hashes)):
                self.embeds_cache[img_hash] = noncached_scores[i]
                ratings[idx] = noncached_scores[i]
        else:
            ratings = noncached_scores
        return ratings

    def __call__(self, img_fps):
        ratings = self.extract_embeds(img_fps)
        return ratings

"""
USAGE EXAMPLE:
import IQA_model_utils as IQA
IQA = IQA.IQAClass("IQAModel") #file path to model
ratings = IQA.__call__(set1_fps) # pass in file paths for each image as a list
# caching functionality added to speed up inference
"""
