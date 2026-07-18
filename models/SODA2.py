import mmd_mill
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False) #4-->16
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()


        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class EncoderC(nn.Module):

    def __init__(self, input_channels, patch_size):
        super(EncoderC, self).__init__()
        self.kernel_dim = 1
        self.feature_dim = input_channels
        self.sz = patch_size
        # Convolution Layer 1 kernel_size = (1, 1, 7), stride = (1, 1, 2), output channels = 24
        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()

        # Residual block 1
        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0),
                               bias=True)  # padding_mode='replicate',
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0),
                               bias=True)  # padding_mode='replicate',
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU()
        # Finish

        # Convolution Layer 2 kernel_size = (1, 1, (self.feature_dim - 6) // 2), output channels = 128
        self.conv4 = nn.Conv3d(24, 192, kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(192)
        self.activation4 = nn.ReLU()

        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        self.bn5 = nn.BatchNorm3d(24)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               bias=True)  # padding_mode='replicate',
        self.bn6 = nn.BatchNorm3d(24)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(24, 96, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               bias=True)  # padding_mode='replicate',
        self.bn7 = nn.BatchNorm3d(96)
        self.activation7 = nn.ReLU()
        self.conv8 = nn.Conv3d(24, 96, kernel_size=1)
        # Finish

        # Combination shape
        self.inter_size = 192 + 96


        # Residual block 3
        self.conv9 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               bias=True)  # padding_mode='replicate',
        self.bn9 = nn.BatchNorm3d(self.inter_size)
        self.activation9 = nn.ReLU()
        self.conv10 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                                bias=True)  # padding_mode='replicate',
        self.bn10 = nn.BatchNorm3d(self.inter_size)
        self.activation10 = nn.ReLU()

        # attention
        self.feature_Select = nn.Sequential(
            nn.Linear(self.inter_size, 128),
            nn.ReLU(),
            nn.Linear(128,2),
            nn.Softmax(dim=1)
        )

        # attention
        self.ca = ChannelAttention(192)#192,288
        self.sa = SpatialAttention()

        # Average pooling kernel_size = (5, 5, 1)
        self.avgpool = nn.AvgPool3d((1, self.sz, self.sz))

        # Fully connected Layer
        # self.fc1 = nn.Linear(in_features=self.inter_size, out_features=n_classes)
        self.__in_features = self.inter_size

        # parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = x.unsqueeze(1)  # (64,1,100,9,9)
        #####光谱特征提取
        # Convolution layer 1
        x1 = self.conv1(x)
        x1 = self.activation1(self.bn1(x1))
        # Residual layer 1
        residual = x1
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))
        x1 = self.conv3(x1)
        x1 = residual + x1  # (32,24,21,7,7)
        x1 = self.activation3(self.bn3(x1))

        # Convolution layer to combine rest
        x1 = self.conv4(x1)  # (32,128,1,7,7)
        x1 = self.activation4(self.bn4(x1))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))  # (64,192,7,7)

        #空间特征提取
        x2 = self.conv5(x)  # (32,24,1,7,7)
        x2 = self.activation5(self.bn5(x2))

        # Residual layer 2
        residual = x2
        residual = self.conv8(residual)  # (32,24,1,7,7)
        x2 = self.conv6(x2)  # (32,24,1,7,7)
        x2 = self.activation6(self.bn6(x2))
        x2 = self.conv7(x2)  # (32,24,1,7,7)
        x2 = residual + x2

        x2 = self.activation7(self.bn7(x2))
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4)) # (64,96,7,7)

        #MFE
        x1 = self.ca(x1) * x1
        x2 = self.sa(x2) * x2
        x = torch.cat((x1, x2), 1)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        """
        #MFE-F
        x = torch.cat((x1, x2), 1)
        x1 = self.ca(x) * x
        x2 = self.sa(x) * x
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        """

        return x1,x2,x

    def output_num(self):
        return self.__in_features

class Decoder(nn.Module):

    def __init__(self, in_dim, out_dim, bottle_neck_dim, bias=True):
        super(Decoder, self).__init__()
        self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        self.fc = nn.Linear(bottle_neck_dim, out_dim, bias=bias)
        self.main = nn.Sequential(
            self.bottleneck,
            nn.Sequential(
                nn.BatchNorm1d(bottle_neck_dim),
                nn.LeakyReLU(0.2, inplace=True),
                self.fc
            )
        )

    def forward(self, x):
        for module in self.main.children():
            x = module(x)
        return x

class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True)
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True)
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

class Diffvalue(nn.Module):
    def __init__(self):
        super(Diffvalue, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        # 将特征展平为 [batch_size, feature_dim]
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # L2归一化（保持每个样本的特征维度）
        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True)  # [batch_size, 1]
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)  # [batch_size, feature_dim]

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True)  # [batch_size, 1]
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)  # [batch_size, feature_dim]

        # 计算每个样本自身的输入1和输入2的内积（而非批次内所有样本的交叉内积）
        # 内积结果形状为 [batch_size]
        sample_dot = torch.sum(input1_l2 * input2_l2, dim=1)  # 逐元素相乘后求和，等价于每个样本的内积

        # 对每个样本的内积求平方，得到每个样本的差异损失
        diffvalue = sample_dot.pow(2)  # 形状为 [batch_size]

        return diffvalue  # 不再取mean，直接返回每个样本的损失值

class OSDA(nn.Module):

    def __init__(self, input_channels, patch_size, in_dim, out_dim, bottle_neck_dim):
        super(OSDA, self).__init__()
        self.EncoderC = EncoderC(input_channels, patch_size)
        self.EncoderS = EncoderC(input_channels, patch_size)
        self.Decoder = Decoder(in_dim, out_dim, bottle_neck_dim, bias=True)

    def forward(self, args, img_s, label_s=None, img_t=None):
        if self.training == True:
            img_s, label_s = img_s.to(args.device), label_s.to(args.device)
            img_t = img_t.to(args.device)
            img_s, label_s = Variable(img_s), Variable(label_s)
            img_t = Variable(img_t)

            feat_s_spe, feat_s_spa, feat_s = self.EncoderC(img_s)
            feat_t_spe, feat_t_spa, feat_t = self.EncoderC(img_t)
            feat_t_spe_special, feat_t_spa_special, feat_t_special = self.EncoderS(img_t)

            out_s = self.Decoder(feat_s)

            loss_mmd_spe = mmd_mill.mmd(feat_s_spe, feat_t_spe)
            loss_mmd_spa = mmd_mill.mmd(feat_s_spa, feat_t_spa)
            loss_mmd = args.d * loss_mmd_spe + args.e * loss_mmd_spa

            diff_loss = DiffLoss()
            loss_f = diff_loss(feat_t, feat_t_special)

            label_s_onehot = nn.functional.one_hot(label_s, num_classes=len(args.source_known_classes))
            label_s_onehot = label_s_onehot * (1 - args.ls_eps)
            label_s_onehot = label_s_onehot + args.ls_eps / (len(args.source_known_classes))
            loss_c = mmd_mill.CrossEntropyLoss(label=label_s_onehot, predict_prob=F.softmax(out_s, dim=1))

            # gentropy_loss = torch.sum(-out_s * torch.log(out_s + 1e-5))

            loss = args.a * loss_c + args.b * loss_mmd + args.c * loss_f

            return loss
        else:
            feat_t_spe, feat_t_spa, feat_t = self.EncoderC(img_s)
            feat_t_spe_special, feat_t_spa_special, feat_t_special = self.EncoderS(img_s)

            out_t = self.Decoder(feat_t)
            out_t = F.softmax(out_t, dim=1)
            diff_calculator = Diffvalue()
            diff = diff_calculator(feat_t, feat_t_special)
            return out_t, diff