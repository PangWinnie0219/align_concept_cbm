import pickle
import time
import os
import json
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


def denormalize(x, mean=IMG_MEAN, std=IMG_STD):
    no_batch_flag = False
    if x.dim() == 3:
        no_batch_flag = True
        x = x.unsqueeze(0)
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    ten = torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)
    if no_batch_flag:
        ten = ten[0]
    return ten


def tensor_to_pil_image(image_tensor, mean=IMG_MEAN, std=IMG_STD):
    """
    Convert a PyTorch image tensor to a Pillow image.

    Args:
        image_tensor (torch.Tensor): A PyTorch image tensor of shape (C, H, W).
        mean (list or tuple, optional): The mean values for each channel (R, G, B). Default is ImageNet mean.
        std (list or tuple, optional): The standard deviation values for each channel (R, G, B). Default is ImageNet std.

    Returns:
        PIL.Image.Image: A Pillow image.
    """
    # Ensure the image tensor is a CPU tensor
    image_tensor = image_tensor.cpu()

    # Unnormalize the tensor
    image_tensor = image_tensor * \
        torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)

    # Clip the values to be in the [0, 1] range
    image_tensor = torch.clamp(image_tensor, 0, 1)

    # Convert the tensor to a NumPy array
    image_np = image_tensor.numpy().transpose(1, 2, 0)

    # Scale the pixel values from [0, 1] to [0, 255]
    image_np = (image_np * 255).astype(np.uint8)

    # Create a Pillow image from the NumPy array
    pillow_image = Image.fromarray(image_np)

    return pillow_image


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

def setup_device(gpu_id):
    # set up GPUS
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if int(gpu_id) == -2 and os.getenv('CUDA_VISIBLE_DEVICES') is not None:
        gpu_id = os.getenv('CUDA_VISIBLE_DEVICES')
    elif int(gpu_id) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"set CUDA_VISIBLE_DEVICES={gpu_id}")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device %s" % device)
    return device


def save_checkpoint(path, model, key="model", other_info={}):
    # save model state dict
    checkpoint = {}
    assert 'model' not in other_info
    checkpoint[key] = model.state_dict()
    checkpoint.update(other_info)
    torch.save(checkpoint, path)
    # print("checkpoint saved at", path)


def resume_model(model, resume, state_dict_key="model"):
    '''
    model:pytorch model
    resume: path to the resume file
    state_dict_key: dict key
    '''
    print("resuming trained weights from %s" % resume)

    checkpoint = torch.load(resume, map_location='cpu')
    if state_dict_key is not None:
        pretrained_dict = checkpoint[state_dict_key]
    else:
        pretrained_dict = checkpoint

    try:
        model.load_state_dict(pretrained_dict)
    except RuntimeError as e:
        print(e)
        print("can't load the all weights due to error above, trying to load part of them!")
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict_use = {}
        pretrained_dict_ignored = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                pretrained_dict_use[k] = v
            else:
                pretrained_dict_ignored[k] = v
        pretrained_dict = pretrained_dict_use
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        print("resumed only", pretrained_dict.keys())
        print("ignored:", pretrained_dict_ignored.keys())

    return model


def advanced_load_state_dict(model, pretrained_dict, remove_key_prefix=""):
    model_dict = model.state_dict()
    # c.f. https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/32
    pretrained_dict_use = {}
    pretrained_dict_ignored = {}
    for k, v in pretrained_dict.items():
        # remove prefix if specified
        if len(remove_key_prefix) > 0 and k.startswith(remove_key_prefix):
            k = k[len(remove_key_prefix):]
        # use only if 1) the pretrained model has the same layer as the current model,
        #  and 2) the shape of tensors matches
        if k in model_dict and v.shape == model_dict[k].shape:
            pretrained_dict_use[k] = v
        else:
            pretrained_dict_ignored[k] = v
    pretrained_dict = pretrained_dict_use
    # 2overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # load the new state dict
    model.load_state_dict(model_dict)
    print("resumed:", pretrained_dict.keys())
    print("ignored:", pretrained_dict_ignored.keys())
    return model


def load_best_model_in_memory(model, model_data_in_memory):
    # Load the model from memory to CPU and then send to GPU
    model_in_cpu = torch.load(model_data_in_memory, map_location="cpu")
    model_data_in_memory.close()
    model.load_state_dict(model_in_cpu)
    model.cuda()
    return model
