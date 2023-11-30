import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvEncoder(nn.Module):
    """
    A simple Convolutional Encoder Model
    """

    def __init__(self):
        super().__init__()
        # self.img_size = img_size
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv5 = nn.Conv2d(512, 768, kernel_size=3, stride=1, padding=1)
        # self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.conv1 = nn.Conv2d(3, 16, (3, 3), padding=(1, 1))
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=(1, 1))
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1))
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d((2, 2))

        self.conv4 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1))
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d((2, 2))

        self.conv5 = nn.Conv2d(128, 256, (3, 3), padding=(1, 1))
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d((2, 2))

    def forward(self, x):
        # Downscale the image with conv maxpool etc.
        # print(x.shape)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # print(x.shape)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # print(x.shape)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        # print(x.shape)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        # print(x.shape)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        # print(x.shape)
        return x


class ConvDecoder(nn.Module):
    """
    A simple Convolutional Decoder Model
    """

    def __init__(self):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(256, 128, (2, 2), stride=(2, 2))
        # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(128, 64, (2, 2), stride=(2, 2))
        # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu2 = nn.ReLU(inplace=True)

        self.deconv3 = nn.ConvTranspose2d(64, 32, (2, 2), stride=(2, 2))
        # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu3 = nn.ReLU(inplace=True)

        self.deconv4 = nn.ConvTranspose2d(32, 16, (2, 2), stride=(2, 2))
        # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu4 = nn.ReLU(inplace=True)

        self.deconv5 = nn.ConvTranspose2d(16, 3, (2, 2), stride=(2, 2))
        # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu5 = nn.ReLU(inplace=True)

    def forward(self, x):
        # print(x.shape)
        x = self.deconv1(x)
        x = self.relu1(x)
        # print(x.shape)

        x = self.deconv2(x)
        x = self.relu2(x)
        # print(x.shape)

        x = self.deconv3(x)
        x = self.relu3(x)
        # print(x.shape)

        x = self.deconv4(x)
        x = self.relu4(x)
        # print(x.shape)

        x = self.deconv5(x)
        x = self.relu5(x)
        # print(x.shape)
        return x



class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv5 = nn.Conv2d(512, 768, kernel_size=3, stride=1, padding=1)
        # self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(3, 16, (3, 3), padding=(1, 1))
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=(1, 1))
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1))
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d((2, 2))

        self.conv4 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1))
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d((2, 2))

        self.conv5 = nn.Conv2d(128, 256, (3, 3), padding=(1, 1))
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d((2, 2))
        
        # Decoder
        # self.t_conv1 = nn.ConvTranspose2d(768, 512, kernel_size=2, stride=2)
        # self.t_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # self.t_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # self.t_conv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # self.t_conv5 = nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)
        self.deconv1 = nn.ConvTranspose2d(256, 128, (2, 2), stride=(2, 2))
        # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(128, 64, (2, 2), stride=(2, 2))
        # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu2 = nn.ReLU(inplace=True)

        self.deconv3 = nn.ConvTranspose2d(64, 32, (2, 2), stride=(2, 2))
        # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu3 = nn.ReLU(inplace=True)

        # self.deconv4 = nn.ConvTranspose2d(32, 16, (2, 2), stride=(2, 2))
        # # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        # self.relu4 = nn.ReLU(inplace=True)

        self.deconv5 = nn.ConvTranspose2d(32, 3, (2, 2), stride=(2, 2))
        # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu5 = nn.ReLU(inplace=True)


    def forward(self, x):
        # Encoder
        # x = F.relu(self.conv1(x))
        # x = self.pool1(x)
        # x = F.relu(self.conv2(x))
        # x = self.pool2(x)
        # x = F.relu(self.conv3(x))
        # x = self.pool3(x)
        # x = F.relu(self.conv4(x))
        # x = self.pool4(x)
        # x = F.relu(self.conv5(x))
        # x = self.pool5(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # print(x.shape)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # print(x.shape)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        # print(x.shape)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        # print(x.shape)

        x = self.conv5(x)
        x = self.relu5(x)
        embeddings = self.maxpool5(x)

        # embeddings = x.mean(dim=(2,3))
        
        # Decoder
        # x = F.relu(self.t_conv1(x))
        # x = F.relu(self.t_conv2(x))
        # x = F.relu(self.t_conv3(x))
        # x = F.relu(self.t_conv4(x))
        # x = torch.sigmoid(self.t_conv5(x))
        # print(x.shape)
        x = self.deconv1(x)
        x = self.relu1(x)
        # print(x.shape)

        x = self.deconv2(x)
        x = self.relu2(x)
        # print(x.shape)

        x = self.deconv3(x)
        x = self.relu3(x)
        # print(x.shape)

        # x = self.deconv4(x)
        # img = self.relu4(x)
        # print(f"SHAPE 1: {x.shape}")

        x = self.deconv5(x)
        img = self.relu5(x)
        print(f"SHAPE 2: {x.shape}")
        
        return embeddings, img


class TransferConvAutoencoder(nn.Module):
    def __init__(self):
        super(TransferConvAutoencoder, self).__init__()
        
        # Encoder
        model = models.__dict__['efficientnet_v2_s'](weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        layers = list(model.children())
        #layers.remove([layers[2]])
        #layers.remove([layers[1]])
        self.new_model = nn.Sequential(*layers[0])

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(1280, 512, (2, 2), stride=(2, 2))
        self.relu1 = nn.ReLU(inplace=True)
        self.t_conv2 = nn.ConvTranspose2d(512, 256, (2, 2), stride=(2, 2))
        self.relu2 = nn.ReLU(inplace=True)
        self.t_conv3 = nn.ConvTranspose2d(256, 128, (2, 2), stride=(2, 2))
        self.relu3 = nn.ReLU(inplace=True)
        self.t_conv4 = nn.ConvTranspose2d(128, 64, (2, 2), stride=(2, 2))
        self.relu4 = nn.ReLU(inplace=True)
        self.t_conv5 = nn.ConvTranspose2d(64, 3, (2, 2), stride=(2, 2))
        self.relu5 = nn.ReLU(inplace=True)

    def forward(self, x):
        # Encoder
        x = self.new_model(x)

        embeddings = x
        
        # Decoder
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = F.relu(self.t_conv4(x))
        x = torch.sigmoid(self.t_conv5(x))
        
        return x, embeddings
