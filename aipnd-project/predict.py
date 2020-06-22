import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import copy
from collections import OrderedDict
import time
from PIL import Image
import json





def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    if image.size[0]<image.size[1]:
        width = 256
        height = width * image.size[1] / image.size[0]
        im = image.resize((width, int(height)), Image.ANTIALIAS)     
    
    else:
        height=256
        width  = height * image.size[0] / image.size[1]
        im = image.resize((int(width), height), Image.ANTIALIAS)
    width, height = im.size   
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    crp = im.crop((left, top, right, bottom))
    np_image = np.array(crp).astype(np.float)
    np_image = np_image/np.array([255,255,255])
    mean,std = np.array([0.485, 0.456, 0.406]),np.array([0.229, 0.224, 0.225])
    val = (np_image-mean)/std
    tp = val.transpose((2,0,1))
    return tp


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, topk,cuda):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    im = Image.open(image_path)
    test = process_image(im)
    te = torch.from_numpy(test).float()
    if cuda:
        te=te.unsqueeze_(0).cuda()
        model.cuda()
    else:
        te=te.unsqueeze_(0)
        model.cpu()
    pre = model.forward(te)
    return torch.exp(pre).topk(topk)



def load_model(path):
    checkpoint = torch.load(path)
    classifier = Network(checkpoint['input_size'],
                    checkpoint['output_size'],
                    checkpoint['hidden_layers'])
    if checkpoint['model']=='vgg16':
        model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.0005)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    input_size,output_size = checkpoint['input_size'],checkpoint['output_size']
    sizes = [input_size,output_size]
    return sizes,model,optimizer,epoch


def visualise(image_path,model,topk,class_no,f,cuda):
    image = Image.open('test/65/image_03211.jpg')
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    probs,classes = predict(image_path,model,5,cuda)
    probs = probs.data.cpu().numpy()[0,:]
    classes= classes.cpu().numpy()
    class_name = np.array([cat_to_name[str(i+1)] for i in classes[0,:]])

    fig, (ax1, ax2) = plt.subplots(figsize=(6,7), nrows=2)
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(cat_to_name[class_no])
    ax2.barh(np.arange(topk), probs)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(topk))
    ax2.set_yticklabels(class_name, size='large');
    ax2.set_xlim(0, 1.1)

def result(image_path,model,topk,class_no,f,cuda):
    #image = Image.open('test/65/image_03211.jpg')
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    probs,classes = predict(image_path,model,5,cuda)
    probs = probs.data.cpu().numpy()[0,:]
    classes= classes.cpu().numpy()
    class_name = np.array([cat_to_name[str(i+1)] for i in classes[0,:]])
    for i in zip(probs,class_name):
        print("{} : {}".format(i[1],i[0]))
        print("\nCorrect Class:{}".format(class_no))
#     fig, (ax1, ax2) = plt.subplots(figsize=(6,7), nrows=2)
#     ax1.imshow(image)
#     ax1.axis('off')
#     ax1.set_title(cat_to_name[class_no])
#     ax2.barh(np.arange(5), probs)
#     ax2.set_aspect(0.1)
#     ax2.set_yticks(np.arange(5))
#     ax2.set_yticklabels(class_name, size='large');
#     ax2.set_xlim(0, 1.1)

    
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--image_path", dest="image_path",required=True,
                        help="provide the directory for image ex: -d data/img1.jpeg")
    parser.add_argument("-f", "--file_path", dest="f",required=True,
                        help="provide the directory for category to name json file ex: -d data/cat.json")
    parser.add_argument("-chkpt", "--checkpoint", dest="checkpoint",required=True,
                        help="provide the directory for saved model checkpoint ex: -d data/chkpt.pth")
    parser.add_argument("-c", "--class", dest="class_no",required=True,type=int,
                        help="provide the correct class no. for the image")
    parser.add_argument("-v", "--visualise", dest="visual",action="store_true",default=False,
                        help="If visualisation of the correct flower and the graph of probs should be provided")
    parser.add_argument("-t", "--topk", dest="topk",default=5,type=int,
                        help="provide the number of probs, defaults to 5")
    parser.add_argument("-gpu", dest="cuda",
                        action="store_true", default=False,
                        help="if gpu is installed, default is False")
    
    
    args = parser.parse_args()
    
    
    sizes,model,optimizer,epoch = load_model(args.checkpoint)
    if args.visual:
        visualise(args.image_path,model,args.topk,args.class_no,args.f,args.cuda)
    else:
        result(args.image_path,model,args.topk,args.class_no,args.f,args.cuda)