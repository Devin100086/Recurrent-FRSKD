import json
import torch
from models import cifarresnet18
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image

resnet = torch.load('./resnet18_entire_model.pth')

image = cv2.imread("image.JPEG")
normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
preprocess = transforms.Compose([transforms.ToPILImage(),transforms.Resize(32), transforms.ToTensor(), normalize, ])

pytorch_tensor = preprocess(image)

pytorch_tensor = pytorch_tensor.to(torch.device("cuda:0"))

outputs = resnet(pytorch_tensor.unsqueeze(0))
_, predicted = torch.max(outputs[1], 1)

print("hello")


