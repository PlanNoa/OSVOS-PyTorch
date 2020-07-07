import numpy as np
import cv2
import torch
import networks.vgg_osvos as vo


files = ["data/DAVIS/JPEGImages/480p/frames/00000.jpg",
         "data/DAVIS/JPEGImages/480p/frames/00001.jpg"]
fname = '00000+00001'

device = torch.device("cpu")

net = vo.OSVOS().eval()
net.load_state_dict(torch.load('models/parent_epoch-239.pth',
                               map_location=lambda storage, loc: storage))
net.to(device)

with torch.no_grad():
    imgs = [cv2.imread(file) for file in files]
    imgs = np.array(imgs, dtype=np.float32)
    imgs = np.array([np.subtract(img, np.array((104.00699, 116.66877, 122.67892), dtype=np.float32)) for img in imgs])
    imgs = imgs.transpose((0, 3, 1, 2))
    imgs = torch.tensor(imgs)

    inputs = imgs.to(device)

    outputs = net.forward(inputs)
    preds = np.transpose(outputs[-1].cpu().data.numpy(), (0, 2, 3, 1))
    preds = [np.squeeze(1 / (1 + np.exp(-pred))) for pred in preds]
    pred = preds[0] + preds[1]
    pred[pred>0.7] = 1
    pred[pred<=0.7] = 0
    print(pred.shape)
