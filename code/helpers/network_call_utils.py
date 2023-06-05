from scipy.special import softmax
import cv2
import numpy as np
from PIL import Image
import torchvision
import torch
from helpers import m
import os
import uuid
import matplotlib.pyplot as plt

def classify_fish(img, model):
    output = model(img.unsqueeze(0))
    pred_np = output.detach().cpu().numpy()
    sm = softmax(pred_np, axis=1)
    c = np.argmax(sm, axis=1)
    cert = np.max(sm, axis=1)
    return c.item(), cert.item()

def torax_imgs_to_box_map(imgs, trck_IDs, model):
    box_map = {}
    for i in range(len(imgs)):
        img_pil = Image.fromarray(imgs[i])
        img_tensor = torchvision.transforms.functional.pil_to_tensor(img_pil)
        img_tensor = torch.div(img_tensor, 255)
        img_tensor = img_tensor.to(m.DEVICE)

        c, cert = classify_fish(img_tensor, model)
        box_map[trck_IDs[i]] = [c, cert]
    return box_map