import os
import random
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch import nn


class GetTransform:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, split):
        if split == "train":
            tforms = [
                transforms.Resize(256),
                transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ]
        elif split == "nocrop":
            tforms = [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ]
        else:
            tforms = [
                transforms.Resize(256),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ]

        return transforms.Compose(tforms)


# https://github.com/pytorch/vision/issues/848
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
get_transforms = GetTransform(mean=IMG_MEAN, std=IMG_STD)


def get_image_encoder(name, pretrained=True):
    if name.startswith('timm.'):
        # timm model zoo
        name = name[5:]  # remove timm.
        model = timm.create_model(name, pretrained=pretrained, num_classes=0)
    else:
        # pytorch model zoo
        weights = "DEFAULT" if pretrained else None
        model = getattr(torchvision.models, name)(weights=weights)
        if name.startswith('vgg'):
            num_features = int(model.classifier[6].in_features)
            model.classifier[6] = nn.Identity()
        elif name.startswith('resnet'):
            model.fc = nn.Identity()
        elif name.startswith('convnext'):
            model.classifier[-1] = nn.Identity()
        elif name.startswith('vit'):
            model.heads = nn.Identity()
        else:
            raise ValueError(f"Unsupported image encoder: {name}")

    # Infer the output size of the image encoder
    with torch.inference_mode():
        out = model(torch.randn(5, 3, 224, 224))
    assert out.dim() == 2
    assert out.size(0) == 5
    image_encoder_output_dim = out.size(1)

    return model, image_encoder_output_dim


def make_deterministic(seed, cudnn_deterministic=True,cudnn_benchmark=False, cudnn_enabled=True):
    # https://github.com/pytorch/pytorch/issues/7068#issuecomment-487907668
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    if cudnn_benchmark:
        print("cudnn benchmark enabled!")
    torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
    if not cudnn_deterministic:
        print("cudnn deterministic disabled, so the results might be different per run")
    torch.backends.cudnn.enabled  = bool(cudnn_enabled)
    if not cudnn_enabled:
        print("cudnn disabled. strict reproducability required? make sure to set num_workers=0 too.")
        print("see https://github.com/pytorch/pytorch/issues/7068#issuecomment-515728600 for details.")


def save_checkpoint(path, model, key="model", other_info={}):
    # save model state dict
    checkpoint = {}
    assert 'model' not in other_info
    checkpoint[key] = model.state_dict()
    checkpoint.update(other_info)
    torch.save(checkpoint, path)
    # print("checkpoint saved at", path)
