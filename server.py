from flask import Flask, jsonify, request
import flask_cors
import base64
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import torchvision.models as models
import io

app = Flask(__name__)
flask_cors.CORS(app)

def process_image(img):
    width, height = img.size
    img = img.resize((224, int(224*(height/width))) if width < height else (int(224*(width/height)), 224))
    width, height = img.size
    img = np.array(img)
    img = img.transpose((2, 0, 1))
    img = img/255
    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225
    img = img[np.newaxis,:]
    image = torch.from_numpy(img)
    image = image.float()
    return image


def predict(image, model):
    output = model.forward(image)    
    _, classes = output.topk(1, dim=1)
    return classes.item()


types = ["Black Polka Dot", "Grey Doctor Who", "Green and Grey", "Blue and Yellow Call Mom", "Red and Black Math", "Light Pink Cartoon"]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        print("INITIALIZED")
        net = models.vgg16(pretrained=True)
        del net.classifier
        self.convs = net.features
        self.pool = nn.AvgPool2d(kernel_size=7)
        self.linear = nn.Sequential(nn.Linear(512, 256),
                                    nn.Sigmoid(),
                                    nn.Linear(256, 64),
                                    nn.Sigmoid(),
                                    nn.Linear(64, 6))
        
        
    def forward(self, x):
        x = self.convs(x)
        x = self.pool(x)
        x = self.linear(x.view(-1)).unsqueeze(0)
        return x


net = CNN()
net.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))

@app.route('/', methods=['POST', 'PUT'])
def post():
    r = request.get_json(force=True)
    sock1io = io.BytesIO(base64.b64decode(r["sock1"][2:-1]))
    sock2io = io.BytesIO(base64.b64decode(r["sock2"][2:-1]))
    sock1 = Image.open(sock1io)
    sock2 = Image.open(sock2io)
    image1 = process_image(sock1)
    image2 = process_image(sock2)
    classification1 = types[predict(image1, net)]
    classification2 = types[predict(image2, net)]
    return jsonify({'type1' : classification1, 'type2' : classification2, 'matches' : (classification1 == classification2)})

@app.route('/', methods=['GET'])
def hw():
    return "hello world"

if __name__ == '__main__':
        app.run(host='0.0.0.0', debug=True)


