import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import argparse
from PIL import Image
import numpy as np
import torchvision.models as models
import os
import random

#our libs
import radam

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        print("INITIALIZED")
        net = models.vgg16()
        net.load_state_dict(torch.load("vgg16.pth"))
        del net.classifier
        self.convs = net.features
        self.pool = nn.AvgPool2d(kernel_size=7)
        self.linear = nn.Sequential(nn.Linear(512, 256),
                                    nn.Sigmoid(),
                                    nn.Linear(256, 64),
                                    nn.Sigmoid(),
                                    nn.Linear(64, 20))
        
        
    def forward(self, x):
        x = self.convs(x)
        x = self.pool(x)
        x = self.linear(x.view(-1)).unsqueeze(0)
        return x

transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_set = datasets.ImageFolder("./data2", transform = transformations)
val_set = datasets.ImageFolder("./data2", transform = transformations)

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 1, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size = 1, shuffle=True)

def optimizer(net, args):
    assert args.optimizer.lower() in ["sgd", "adam", "radam"], "Invalid Optimizer"

    if args.optimizer.lower() == "sgd":
	       return optim.SGD(net.parameters(), lr=args.lr, momentum=args.beta1, nesterov=args.nesterov)
    elif args.optimizer.lower() == "adam":
	       return optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optimizer.lower() == "radam":
            return radam.RAdam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))


def process_image(image_path):
    # Load Image
    img = Image.open(image_path)

    # Get the dimensions of the image
    width, height = img.size

    # Resize by keeping the aspect ratio, but changing the dimension
    # so the shortest size is 224px
    img = img.resize((224, 224))
    # Turn image into numpy array
    img = np.array(img)

    # Make the color channel dimension first instead of last
    img = img.transpose((2, 0, 1))

    # Make all values between 0 and 1
    img = img/255

    # Normalize based on the preset mean and standard deviation
    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225

    # Add a fourth dimension to the beginning to indicate batch size
    img = img[np.newaxis,:]

    # Turn into a torch tensor
    image = torch.from_numpy(img)
    image = image.cuda().float()
    return image


def predict(image, model):
    output = model.forward(image)
    
    probs, classes = output.topk(1, dim=1)

    return classes.item()

def test(net, criterion, testloader, args):
    net.eval()
    with torch.no_grad():
        correct = 0
        os.chdir("data2/")
        folders = os.listdir()
        os.chdir("1")
        for folder in folders:
            os.chdir("../"+folder)
            for i in range(0, 2):
                pics = os.listdir()
                pic = random.choice(pics)
                image = process_image(pic)
                if int(predict(image, net)) == int(folder):
                    correct += 1
        print("Test set accuracy: " + str(float(correct)/ float(len(folders*3))))
        os.chdir("../../")



def train(net, criterion, optimizer, trainloader, args):
    for epoch in range(1, args.epoch+1):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            inputs=inputs.cuda()
            outputs = net(inputs)
            labels = labels.cuda()
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 0 and i > 0:
                print('[Epoch %02d, Minibatch %05d] Loss: %.5f' %
			    (epoch, i, running_loss/10))
                running_loss = 0.0
        if epoch % 2 == 0 and epoch > 0:
            torch.save(net.state_dict(), "cnn.pt")
            test(net, criterion, testloader, args)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # optimization related arguments
    parser.add_argument('--batch_size', default=1, type=int,
                        help='input batch size')
    parser.add_argument('--epoch', default=100, type=int,
                        help='epochs to train for')
    parser.add_argument('--optimizer', default='radam', help='optimizer')
    parser.add_argument('--lr', default=0.000001, type=float, help='LR')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--nesterov', default=False)

    args = parser.parse_args()

    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    #data loader
    #trainloader, testloader = dataloader(args)
    trainloader = train_loader
    testloader = val_loader

    
    net = CNN()
    net.cuda()
    '''
    net = models.vgg16()
    net.load_state_dict(torch.load("vgg16.pth"))
    print(net)
    
    netPool = net.avgpool
    t = nn.Sequential()
    t.add_module("32", nn.Linear(25088, 4096))
    t.add_module("33", nn.ReLU(inplace=True))
    t.add_module("34", nn.Dropout(p=0.5, inplace=False))
    t.add_module("35", nn.Linear(4096, 1024))
    t.add_module("36", nn.ReLU(inplace=True))
    t.add_module("37", nn.Dropout(p=0.5, inplace=False))
    t.add_module("38", nn.Linear(1024, 256))
    t.add_module("39", nn.ReLU(inplace=True))
    t.add_module("40", nn.Dropout(p=0.5, inplace=False))
    t.add_module("41", nn.Linear(256, 80))
    t.add_module("42", nn.ReLU(inplace=True))
    t.add_module("43", nn.Dropout(p=0.5, inplace=False))
    t.add_module("44", nn.Linear(80, 20))
    
    net.classifier = t
    net.cuda()
    print(net)
    print("Loaded")
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer(net, args)
    #optimizer = optim.Adam(net.parameters(), eps=1e-07)
    #optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    #net.load_state_dict(torch.load("cnn1.pt"))
    train(net, criterion, optimizer, trainloader, args)
    
    
    print("Training completed!")
