# reference: https://www.zhihu.com/question/272988870/answer/562262315

#original version
作者：AlexL
链接：https://www.zhihu.com/question/272988870/answer/562262315
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

def conv_block(neurons, block_input, bn=False, dropout=None):
    conv1 = Conv2D(neurons, (3,3), padding='same', kernel_initializer='glorot_normal')(block_input)
    if bn:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    if dropout is not None:
        conv1 = SpatialDropout2D(dropout)(conv1)
    conv2 = Conv2D(neurons, (3,3), padding='same', kernel_initializer='glorot_normal')(conv1)
    if bn:
        conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    if dropout is not None:
        conv2 = SpatialDropout2D(dropout)(conv2)
    pool = MaxPooling2D((2,2))(conv2)
    return pool, conv2  # returns the block output and the shortcut to use in the uppooling blocks

def middle_block(neurons, block_input, bn=False, dropout=None):
    conv1 = Conv2D(neurons, (3,3), padding='same', kernel_initializer='glorot_normal')(block_input)
    if bn:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    if dropout is not None:
        conv1 = SpatialDropout2D(dropout)(conv1)
    conv2 = Conv2D(neurons, (3,3), padding='same', kernel_initializer='glorot_normal')(conv1)
    if bn:
        conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    if dropout is not None:
        conv2 = SpatialDropout2D(dropout)(conv2)
    
    return conv2

def deconv_block(neurons, block_input, shortcut, bn=False, dropout=None):
    deconv = Conv2DTranspose(neurons, (3, 3), strides=(2, 2), padding="same")(block_input)
    uconv = concatenate([deconv, shortcut])
    uconv = Conv2D(neurons, (3, 3), padding="same", kernel_initializer='glorot_normal')(uconv)
    if bn:
        uconv = BatchNormalization()(uconv)
    uconv = Activation('relu')(uconv)
    if dropout is not None:
        uconv = SpatialDropout2D(dropout)(uconv)
    uconv = Conv2D(neurons, (3, 3), padding="same", kernel_initializer='glorot_normal')(uconv)
    if bn:
        uconv = BatchNormalization()(uconv)
    uconv = Activation('relu')(uconv)
    if dropout is not None:
        uconv = SpatialDropout2D(dropout)(uconv)
        
    return uconv

def build_model(start_neurons, bn=False, dropout=None):    
    input_layer = Input((128, 128, 1))
    # 128 -> 64
    conv1, shortcut1 = conv_block(start_neurons, input_layer, bn, dropout)
    # 64 -> 32
    conv2, shortcut2 = conv_block(start_neurons * 2, conv1, bn, dropout)
    # 32 -> 16
    conv3, shortcut3 = conv_block(start_neurons * 4, conv2, bn, dropout)
    # 16 -> 8
    conv4, shortcut4 = conv_block(start_neurons * 8, conv3, bn, dropout)   
    #Middle
    convm = middle_block(start_neurons * 16, conv4, bn, dropout)   
    # 8 -> 16
    deconv4 = deconv_block(start_neurons * 8, convm, shortcut4, bn, dropout)  
    # 16 -> 32
    deconv3 = deconv_block(start_neurons * 4, deconv4, shortcut3, bn, dropout)   
    # 32 -> 64
    deconv2 = deconv_block(start_neurons * 2, deconv3, shortcut2, bn, dropout)
    # 64 -> 128
    deconv1 = deconv_block(start_neurons, deconv2, shortcut1, bn, dropout)  
    #uconv1 = Dropout(0.5)(uconv1)
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(deconv1) 
    model = Model(input_layer, output_layer)
    return model
    
#modify version
# instead of transposed convolution,we choose upsampling+3*3 conv for better performance

"""
把encoder替换预训练的模型的诀窍在于，如何很好的提取出pretrained models在不同尺度上提取出来的信息，
并且如何把它们高效的接在decoder上。
常见的用于嫁接的模型有Inception和Mobilenet，但我在这里就分析一下更直观一些的ResNet/ResNeXt这一类的模型：
"""

def __init__(self):
        super().__init__()
        self.resnet = models.resnet34(pretrained=True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )

        self.encoder2 = self.resnet.layer1 # 64
        self.encoder3 = self.resnet.layer2 #128
        self.encoder4 = self.resnet.layer3 #256
        self.encoder5 = self.resnet.layer4 #512

        self.center = nn.Sequential(
            ConvBn2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.decoder5 = Decoder(256+512,512,64)
        self.decoder4 = Decoder(64 +256,256,64)
        self.decoder3 = Decoder(64 +128,128,64)
        self.decoder2 = Decoder(64 +64 ,64 ,64)
        self.decoder1 = Decoder(64     ,32 ,64)

        self.logit = nn.Sequential(
            nn.Conv2d(384, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        mean=[0.485, 0.456, 0.406]
        std=[0.229,0.224,0.225]
        x=torch.cat([
           (x-mean[2])/std[2],
           (x-mean[1])/std[1],
           (x-mean[0])/std[0],
        ],1)

        e1 = self.conv1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        f = self.center(e5)
        d5 = self.decoder5(f, e5)
        d4 = self.decoder4(d5,e4)
        d3 = self.decoder3(d4,e3)
        d2 = self.decoder2(d3,e2)
        d1 = self.decoder1(d2)

"""
关于decoder的设计方法，还有两个可以参考的小技巧：
一是 Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks，
可以理解为是一种attention，用很少的参数来校准feature map，详情请见论文，但实现细节可参考以下的PyTorch代码：
"""

作者：AlexL
链接：https://www.zhihu.com/question/272988870/answer/562262315
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

class sSE(nn.Module):
    def __init__(self, out_channels):
        super(sSE, self).__init__()
        self.conv = ConvBn2d(in_channels=out_channels,out_channels=1,kernel_size=1,padding=0)
    def forward(self,x):
        x=self.conv(x)
        #print('spatial',x.size())
        x=F.sigmoid(x)
        return x

class cSE(nn.Module):
    def __init__(self, out_channels):
        super(cSE, self).__init__()
        self.conv1 = ConvBn2d(in_channels=out_channels,out_channels=int(out_channels/2),kernel_size=1,padding=0)
        self.conv2 = ConvBn2d(in_channels=int(out_channels/2),out_channels=out_channels,kernel_size=1,padding=0)
    def forward(self,x):
        x=nn.AvgPool2d(x.size()[2:])(x)
        #print('channel',x.size())
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.sigmoid(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = ConvBn2d(in_channels, channels, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(channels, out_channels, kernel_size=3, padding=1)
        self.spatial_gate = sSE(out_channels)
        self.channel_gate = cSE(out_channels)

    def forward(self, x, e=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        #print('x',x.size())
        #print('e',e.size())
        if e is not None:
            x = torch.cat([x,e],1)

        x = F.relu(self.conv1(x),inplace=True)
        x = F.relu(self.conv2(x),inplace=True)
        #print('x_new',x.size())
        g1 = self.spatial_gate(x)
        #print('g1',g1.size())
        g2 = self.channel_gate(x)
        #print('g2',g2.size())
        x = g1*x + g2*x
        return x
#moreover
"""
还有一个就是为了进一步鼓励模型在多尺度上的鲁棒性，我们可以引入Hypercolumn去直接把各个scale的feature map concatenate起来：
"""
f = torch.cat((
            F.upsample(e1,scale_factor= 2, mode='bilinear',align_corners=False),
            d1,
            F.upsample(d2,scale_factor= 2, mode='bilinear',align_corners=False),
            F.upsample(d3,scale_factor= 4, mode='bilinear',align_corners=False),
            F.upsample(d4,scale_factor= 8, mode='bilinear',align_corners=False),
            F.upsample(d5,scale_factor=16, mode='bilinear',align_corners=False),
        ),1)

f = F.dropout2d(f,p=0.50)
logit = self.logit(f)
"""
更神奇的方法就是直接把每个scale的feature map和downsized gt进行比较计算loss，最后各个尺度的loss进行加权平均。
详情请见这里的讨论：Deep semi-supervised learning | Kaggle 这里就不再赘述了。

"""

