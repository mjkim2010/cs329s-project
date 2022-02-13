import torch
from torch import nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
import joblib


class ImageDataset(Dataset):
    def __init__(self, imgs=None, img_fps=None):
        if imgs is not None:
            self.imgs = imgs
        elif img_fps is not None:
            self.img_fps = img_fps
        else:
            raise ValueError('Either imgs or img_fps must be non-null')
        self.norm = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        input_img = self.imgs[idx] if hasattr(self, 'imgs') else Image.open(self.img_fps[idx])
        if input_img.shape[0] != 3: # handle greyscale images
            input_img = torch.cat((input_img, input_img, input_img), dim=0)
        input_img = self.norm(input_img)
        return input_img

    def __len__(self):
        return len(self.imgs)


def load_models(clf_fp=None):
    model = torchvision.models.vgg16(pretrained=True)
    modules = list(model.children())[:-1]
    modules.append(nn.AvgPool2d(7, 1))
    class Flatten(nn.Module):
        def forward(self, input):
            return input.view(input.size(0), -1)
    flatten = Flatten()
    modules.append(flatten)
    model = nn.Sequential(*modules)
    if torch.cuda.is_available():
        model = model.cuda()
    clf = joblib.load("MVP_IQA_SE_L2_0.00001_ES.pkl" if clf_fp is None else clf_fp)
    return model, clf

def score_image(model, clf, images):
    embeds = model(images)
    preds = clf.predict(embeds)
    return preds