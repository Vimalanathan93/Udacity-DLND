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

# get dataset path
# get network parameters -hidden units,learning rate, epochs
# get model parameters -vgg16 or resnet50
# get if cuda
# training and validation log

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)

def get_data(path):
    data_dir = path
    train_dir = data_dir + 'train'
    valid_dir = data_dir + 'val'
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }


    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_iter = torch.utils.data.DataLoader(image_datasets['train'], batch_size=32,shuffle=True, num_workers=4)
    val_iter = torch.utils.data.DataLoader(image_datasets['val'], batch_size=16,shuffle=True, num_workers=4)
    
    return train_iter,val_iter,image_datasets


def train(train_iter,val_iter,pretrainmodel,hidden_units,lr,drop_p,cuda,epochs,print_every,verbose):
    
    
    
    if pretrainmodel=='resnet18':
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = Network(512,102,hidden_units,drop_p)
        model.fc = classifier
    else:
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = Network(25088,102,hidden_units,drop_p)
        model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    
    steps = 0
    running_loss = 0
    for e in range(epochs):
        # Model in training mode, dropout is on
        start = time.time()
        if cuda:
            model.cuda()
        model.train()
        for images, labels in iter(train_iter):
            steps += 1
            start1=time.time()
            inputs = Variable(images)
            targets = Variable(labels)
            if cuda:
                inputs,targets = inputs.cuda(),targets.cuda()
            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            if steps % print_every == 0:
                model.eval()
                accuracy = 0
                val_loss = 0
                for ii, (images, labels) in enumerate(val_iter):
                    inputs = Variable(images, volatile=True)
                    labels = Variable(labels, volatile=True)
                    if cuda:
                        inputs,labels = inputs.cuda(),labels.cuda()
                    output = model.forward(inputs)
                    val_loss += criterion(output, labels).data[0]
                    ps = torch.exp(output).data
                    equality = (labels.data == ps.max(1)[1])
                    accuracy += equality.type_as(torch.FloatTensor()).mean()
                if verbose==1:
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Val Loss: {:.3f}.. ".format(val_loss/len(val_iter)),
                          "Val Accuracy: {:.3f}..".format(accuracy/len(val_iter)),
                          "Time taken for {} steps: {:.3f} seconds..".format(print_every,time.time()-start1)
                         )        
                running_loss = 0
                model.train()
        if verbose==1:
            print("Time taken for {} epoch : {:.3f} seconds..".format(e+1,time.time()-start))
        
        return model,optimizer,classifier
        
def store_checkpoint(filepath,model,optimizer,classifier):
    model.class_to_idx = image_datasets['train'].class_to_idx
    if pretrainmodel=='vgg16':
        input_size = 25088
        hidden_layers = [each.out_features for each in model.classifier.hidden_layers]
    else:
        input_size=512
        hidden_layers = [each.out_features for each in classifier.hidden_layers]
    checkpoint = {'input_size': input_size,
              'output_size': output_size,
              'hidden_layers': hidden_layers,
              'state_dict': model.state_dict(),
              'epochs':epochs,
              'optimizer_state_dict':optimizer.state_dict,
              'model':'vgg16'
             }

    torch.save(checkpoint, filepath)



if __name__ == "__main__":
    import argparse



    # get dataset path
    # get network parameters -hidden units,learning rate, epochs
    # get model parameters -vgg16 or resnet50
    # get if cuda
    # training and validation log



    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", dest="filepath",required=True,
                        help="provide the directory for dataset ex: -d data/flowers")
    parser.add_argument("-gpu", dest="cuda",
                        action="store_true", default=False,
                        help="if gpu is installed, default is False")
    
    parser.add_argument("-hidden",nargs='+',dest="hidden",
                        required=True, default=False,
                        help="provide the hidden unit sizes separated by spaces ex: 4096 2000 512")
    
    parser.add_argument("-lr",type=float,dest="lr",
                        default=0.0005,
                        help="provide learning rate, defaults to 0.0005")
    parser.add_argument("-epochs",type=int,dest="epochs",
                        required=True, 
                        help="provide number of epochs")
    parser.add_argument("-pretrainmodel",dest="pretrainmodel",
                        default="vgg16",
                        help="provide the pretrained model to use: either vgg16 or resnet18")
    parser.add_argument("-drop_p",dest="drop_p",
                        default=0.5,type=float,
                        help="Provide dropout probability, defaults to 0.5")
#     parser.add_argument("-out_s",dest="out_s",
#                         default=102,type=int,
#                         help="Provide output size for the model")
    parser.add_argument("-verbose",dest="verbose",
                        default=1,type=int,
                        help="Either 0- do not print training and val logs, 1- print training and val logs")
    parser.add_argument("-chkp",dest="checkpoint",
                        required=True,
                        help="Provide Checkpoint file path")
#     parser.add_argument("-help", "--docs", 
#                         help='''  "-d", "--dataset" : "provide the directory for dataset ex: -d data/flowers"
#                                    "-gpu" : "if gpu is installed, default is False"
#                                    "-hidden": "provide the hidden unit sizes separated by spaces ex: 4096 2000 512"
#                                    "-lr" : "provide learning rate, defaults to 0.0005"
#                                    "-epochs" : "provide number of epochs"
#                                    "-pretrainmodel" : "provide the pretrained model to use: either vgg16 or resnet18"
#                                    "-drop_p" : "Provide dropout probability, defaults to 0.5"
#                                    "-chkp" : "Provide Checkpoint file path"
#                                    "-verbose" : "Either 0- do not print training and val logs, 1- print training and val logs"
#                                    ''')
    
    
    
    
    args = parser.parse_args()
    hidden = [int(i) for i in args.hidden]
  
    train_iter,val_iter,image_datasets = get_data(args.filepath)
    model,optimizer,classifier = train(train_iter,val_iter,args.pretrainmodel,hidden,args.lr,args.drop_p,args.cuda,int(args.epochs),100,args.verbose)
    store_checkpoint(args.checkpoint,model,optimizer,classifier)
    

