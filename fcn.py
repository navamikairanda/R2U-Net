import torch.nn as nn
import torchvision.models.vgg as vgg

class Segnet(nn.Module):
  '''
  Fully Convolutional Network (FCN)
  
  Performs fcn on the inputs and returns the feature map. This section is a decoder of the fcn network which starts from the 7th layer of pretrained vgg model.
  Args:
    n_classes: number of classes to be predicted
    
  Returns:
    feature map size=(N, n_class, x.H/1, x.W/1)
  
  
  '''
  def __init__(self, n_classes):
    super(Segnet, self).__init__()
    self.vgg_model = vgg.vgg16(pretrained=True, progress=True)#.to(device)
    #del self.vgg_model.classifier
    self.relu    = nn.ReLU(inplace=True)
    self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
    self.bn1     = nn.BatchNorm2d(512) 
    self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
    self.bn2     = nn.BatchNorm2d(256)
    self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
    self.bn3     = nn.BatchNorm2d(128)
    self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
    self.bn4     = nn.BatchNorm2d(64)
    self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
    self.bn5     = nn.BatchNorm2d(32)
    self.classifier = nn.Conv2d(32, n_classes, kernel_size=1)

  def forward(self, x):
    x = self.vgg_model.features(x) # B, 
    output = self.vgg_model.avgpool(x) # B, 512, 512, 7
    score = self.bn1(self.relu(self.deconv1(x)))     # size=(N, 512, x.H/16, x.W/16)
    score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
    score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
    score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
    score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
    score = self.classifier(score)                    # size=(N, n_classes, x.H/1, x.W/1)
    return score  # size=(N, n_class, x.H/1, x.W/1)
