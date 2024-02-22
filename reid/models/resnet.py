# TODO: 
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

class Bottleneck(nn.Module):
    """Initializes model with pretrained weights.
    """
    
    expansion = 4

    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        # downsample if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        # add identity
        out += identity
        out = self.relu(out)
        
        return out

      
class ResNet(nn.Module):

    def __init__(self, block, layer_list, num_classes, num_channels=3, last_conv_stride=2, fc_dims=None, dropout_p=None, triplet_loss=False):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.in_channels = 64
        self.feature_dim = 512 * block.expansion
        self.triplet_loss = triplet_loss

        # 3×256×128  -->  64×128×64
        self.conv1 = nn.Conv2d(num_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        # 64×128×64  -->  64×64×32
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 64×64×32  -->  2048×16x4
        self.layer1 = self._make_layer(block, layer_list[0], 64)
        self.layer2 = self._make_layer(block, layer_list[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layer_list[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layer_list[3], 512, stride=last_conv_stride)
        
        # 2048×16x4  -->  2048×1x1
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        # 2048×1x1  --> num_classes 
        self.fc = self._construct_fc_layer(fc_dims, self.feature_dim, dropout_p=dropout_p, bias=False)
        self.classifier = nn.Linear(self.feature_dim, self.num_classes, bias=False)

        # self.fc.apply(weights_init_kaiming)
        # self.classifier.apply(weights_init_classifier)
        self._init_params()

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.gap(x)

        # BNNeck from: https://arxiv.org/abs/1903.07071

        f_t = torch.flatten(x, 1)
        f_i = f_t

        if self.fc is not None:
            f_i = self.fc(f_t)
      
        if not self.training:
            return f_i

        p = self.classifier(f_i)

        if self.triplet_loss:
            return p, f_t
        
        return p


    def _make_layer(self, block, blocks, planes, stride=1):
        """Creates the fully connected BN layers as in https://arxiv.org/abs/1903.07071
        """
        downsample = None
        
        if stride != 1 or self.in_channels != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            )
            
        layers = []
        layers.append(block(self.in_channels, planes, downsample=downsample, stride=stride))
        self.in_channels = planes*block.expansion
        
        for i in range(1, blocks):
            layers.append(block(self.in_channels, planes))
            
        return nn.Sequential(*layers)


    def _init_params(self):
        """Initializes the model parameters.
        """

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None, bias=True):
        """Constructs fully connected layer
        """

        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(
            fc_dims, (list, tuple)
        ), 'fc_dims must be either list or tuple, but got {}'.format(
            type(fc_dims)
        )

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim, bias=bias))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.
    """

    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    """

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model

