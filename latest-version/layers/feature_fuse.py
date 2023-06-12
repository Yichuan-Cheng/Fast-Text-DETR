import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleFeatureSelection(nn.Module):
    def __init__(self, in_channels = 256, inter_channels = 64 , out_features_num=4, attention_type='scale_channel_spatial'):
        super(ScaleFeatureSelection, self).__init__()
        self.in_channels=in_channels
        self.inter_channels = inter_channels
        inner_channels = inter_channels
        self.out_features_num = out_features_num
        bias = False

        self.output_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=bias),
                            nn.GroupNorm(32, in_channels),
                            nn.ReLU(inplace=True)) for i in range(4)])

        self.lateral_conv_4 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=bias),
                                            nn.GroupNorm(32, in_channels))
        self.lateral_conv_3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=bias),
                                            nn.GroupNorm(32, in_channels))
        self.lateral_conv_2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=bias),
                                            nn.GroupNorm(32, in_channels))
        self.channel_mapping = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=bias),
                                            nn.GroupNorm(32, in_channels))
        
        for i in range(4):
            self.output_convs[i].apply(self._initialize_weights)
        # self.out4.apply(self._initialize_weights)
        # self.out3.apply(self._initialize_weights)
        # self.out2.apply(self._initialize_weights)
        self.lateral_conv_4.apply(self._initialize_weights)
        self.lateral_conv_3.apply(self._initialize_weights)
        self.lateral_conv_2.apply(self._initialize_weights)
        self.channel_mapping.apply(self._initialize_weights)



    def _initialize_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)


    def forward(self, encoder_features, one_fourth_feature):
        c2, c3, c4, c5 = encoder_features

        # Basic FPN
        out4 = F.interpolate(c5, size=c4.shape[-2:], mode="bilinear", align_corners=False) + self.lateral_conv_4(c4)  # 1/16
        out3 = F.interpolate(out4, size=c3.shape[-2:], mode="bilinear", align_corners=False) + self.lateral_conv_3(c3)  # 1/8
        out2 = F.interpolate(out3, size=c2.shape[-2:], mode="bilinear", align_corners=False) + self.lateral_conv_2(c2)  # 1/4
        p5 = self.output_convs[0](c5)
        p4 = self.output_convs[1](out4)
        p3 = self.output_convs[2](out3)
        p2 = self.output_convs[3](out2)

        multiscale_feature = [p2, p3, p4, p5]

        mask_feature = F.interpolate(p2, size=one_fourth_feature.shape[-2:] ,mode="bilinear", align_corners=False) \
                       + self.channel_mapping(one_fourth_feature)

        return mask_feature, multiscale_feature