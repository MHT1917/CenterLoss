import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterLoss(nn.Module):

    def __init__(self,cls_num,feature_num):
        super().__init__()
        self.cls_num = cls_num
        self.center = nn.Parameter(torch.randn(cls_num,feature_num))

    def forward(self,xs,ys):
        # xs = F.normalize(xs)
        center_exp = self.center.index_select(dim=0,index=ys.long())
        count = torch.histc(ys,bins=self.cls_num,min=0,max=self.cls_num-1)
        count_exp = count.index_select(dim=0,index=ys.long())
        return torch.sum(torch.div(torch.sum(torch.pow(xs-center_exp,2),dim=1),count_exp.float()))*0.5


class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 32, 3, 1),
            nn.LeakyReLU()
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(32 * 3 * 3, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(2, 10)
        )

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x.view(-1, 32 * 3 * 3))
        out = self.output_layer(x)
        return out,x






