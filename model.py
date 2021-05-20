import torch
import torch.nn as nn
import torchvision



class Plantmodel(nn.Module):
    
    def __init__(self,num_classes=12):
        super().__init__()
        self.model=torchvision.models.resnet18(pretrained=False)
        in_features=self.model.fc.in_features
        self.model.fc=nn.Linear(in_features,num_classes)
    def forward(self,x):
        batch_size, C, H, W = x.shape
        x=self.model(x)
       # x=self.backbone.maxpool(x)
        
       # x=self.backbone.layer1(x)
       # x=self.backbone.layer2(x)
       # x=self.backbone.layer3(x)
       # x=self.backbone.layer4(x)
        
       # x=F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
       # x=F.dropout(x,0.25,self.training)
       # x=self.classifier(x)
        return x
        