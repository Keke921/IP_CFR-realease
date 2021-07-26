import torch.nn as nn
import torch.nn.functional as F
import torch
#from torchvision.models import vgg19
from ResidualBlock import *
import math

class GeneratorRes(nn.Module):
    def __init__(self, channels=3, img_size = 128, num_classes = 10177):
        super(GeneratorRes, self).__init__()

        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(ResidualBlock(out_feat , activation = nn.LeakyReLU() ))
            return layers
        
        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers
         
        self.down_size = img_size//32 
        num_filters = 512
        self.num_filters = num_filters
        down_dim = self.num_filters * (img_size//32) ** 2              # dim of feature: down_dim(2048)
        
        # ----------------
        # net
        # ----------------
        # encoder
          # input 128*128*3
        self.conv0 = nn.Sequential(*downsample(channels, 64)  )            #64*64*64      
        self.conv1 = nn.Sequential(*downsample(64, 64) )                   #32*32*64      
        self.conv2 = nn.Sequential(*downsample(64, 128))                   #16*16*128     
        self.conv3 = nn.Sequential(*downsample(128, 256))                  #8*8*256        
        self.conv4 = nn.Sequential(*downsample(256, num_filters))          #4*4*512        
        
        # extract hide layer
        self.fc1 = nn.Linear(down_dim, num_filters)                        #512
        self.fc2 = nn.MaxPool1d(2, 2, 0)                                   #256
                                                
        #self.predict = nn.Sequential(nn.Dropout(0.3),
        #                             nn.Linear(num_filters//2, num_classes) )
        
        # decoder
        #       
        self.deconv4 = nn.Sequential(*upsample(256+100, 256),          #2*2*256
                                     *upsample(256, 256))             #4*4*256
        self.deconv3 = nn.Sequential(*upsample(256,128) )             #8*8*128       
        self.deconv2 = nn.Sequential(*upsample(128, 64) )             #16*16*64         
        self.deconv1 = nn.Sequential(*upsample(64,  32) )             #32*32*32         
        self.deconv0 = nn.Sequential(*upsample(32,  16) )             #64*64*16          
        
        # 4
        self.Resblk4 = ResidualBlock(768, 768, 2, 1,                  #dim of (deconv4 + conv4) 
                                     padding = [1,0,1,0], 
                                     activation = nn.LeakyReLU() )    
        self.recons4 = nn.Sequential(*[ResidualBlock(768, 768, 2, 1,  #dim of Resblk4 4*4*512  
                                                     padding = [1,0,1,0], 
                                                     activation = nn.LeakyReLU() ) for _ in range(2)]  )
        # 3
        self.reconst_deconv3 = nn.Sequential(*upsample(768, 512) )    #dim of recons4 4*4*768 -> 8*8*512  
        self.Resblk3 = ResidualBlock(384,                             #dim of (deconv3 + conv3) 384
                                     activation = nn.LeakyReLU() )    
        self.recons3 = nn.Sequential(*[ResidualBlock(896,             #dim of (Resblk3 + reconstruct_deconv3) 384+512 
                                                     activation = nn.LeakyReLU() ) for _ in range(2)]  )
        # 2
        self.reconst_deconv2 = nn.Sequential(*upsample(896, 256) )     #dim of recons3 8*8*1024 -> 16*16*256
        self.Resblk2 = ResidualBlock(192,                              #dim of (deconv2 + conv2)  192
                                     activation = nn.LeakyReLU() )    
        self.recons2 = nn.Sequential(*[ResidualBlock(448,              #dim of (Resblk2 + reconstruct_deconv2) 192+256
                                                     activation = nn.LeakyReLU() ) for _ in range(2)]  )        
        # 1
        self.reconst_deconv1 = nn.Sequential(*upsample(448, 128) )     #dim of recons2 16*16*512 -> 32*32*128
        self.Resblk1 = ResidualBlock(96,                               #dim of (deconv1 + conv1)    96 
                                     activation = nn.LeakyReLU() )    
        self.recons1 = nn.Sequential(*[ResidualBlock(224,              #dim of (Resblk1 + reconstruct_deconv1) 96+128
                                                     activation = nn.LeakyReLU() ) for _ in range(2)]  )  
        # 0
        self.reconst_deconv0 = nn.Sequential(*upsample(224, 64) )      #dim of recons1 32*32*224 -> 64*64*64
        self.Resblk0 = ResidualBlock(80,                               #dim of (deconv0 + conv0)  16+64
                                     activation = nn.LeakyReLU() )    
        self.recons0 = nn.Sequential(*[ResidualBlock(144,               #dim of (Resblk0 + reconstruct_deconv0) 
                                                     activation = nn.LeakyReLU() ) for _ in range(2)]  ) 
        # decode to image
        self.upsampling = nn.Sequential(*upsample(144, 32), ResidualBlock(32, activation = nn.LeakyReLU()),
                                     nn.Conv2d(32, channels, 3, 1, 1) )

        
    def forward(self, x, z):
        # encoder
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
       
        # feature
        fc1 = self.fc1(conv4.view(conv4.size(0), -1)) #4*4*512  ->512
        features = self.fc2(fc1.view( fc1.size()[0] , -1 , 2  )).view( fc1.size()[0] , -1 ) 
        #validity = self.predict(features)          
        
        #decocer
        deconv4 = self.deconv4(torch.cat([features,z], 1).view(features.size()[0], -1, 1, 1)) #加入z 增加模式*****
        deconv3 = self.deconv3(deconv4)
        deconv2 = self.deconv2(deconv3)
        deconv1 = self.deconv1(deconv2)
        deconv0 = self.deconv0(deconv1)

        # 4
        Resblk4 = self.Resblk4(torch.cat([deconv4, conv4], 1)) 
        recons4 = self.recons4(Resblk4)        
        # 3                                    
        reconst_deconv3 = self.reconst_deconv3(recons4)        #\
        Resblk3 = self.Resblk3(torch.cat([deconv3, conv3], 1)) #\    
        recons3 = self.recons3(torch.cat([Resblk3, reconst_deconv3], 1)) 
        # 2 
        reconst_deconv2 = self.reconst_deconv2(recons3)        #\
        Resblk2 = self.Resblk2(torch.cat([deconv2, conv2], 1)) #\      
        recons2 = self.recons2(torch.cat([Resblk2, reconst_deconv2], 1)) 
        # 1 
        reconst_deconv1 = self.reconst_deconv1(recons2)        #\
        Resblk1 = self.Resblk1(torch.cat([deconv1, conv1], 1)) #\      
        recons1 = self.recons1(torch.cat([Resblk1, reconst_deconv1], 1))    
        # 0 
        reconst_deconv0 = self.reconst_deconv0(recons1)        #\
        Resblk0 = self.Resblk0(torch.cat([deconv0, conv0], 1)) #\      
        recons0 = self.recons0(torch.cat([Resblk0, reconst_deconv0], 1))                                   
        
        out = self.upsampling(recons0)
                                     
        return features, out   #validity, 
    
    
class Generator(nn.Module):
    def __init__(self, channels=3, img_size = 128, num_classes = 10177):
        super(Generator, self).__init__()
        
        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers
        
        '''
        self.model = nn.Sequential(                       #128*128*3
            *downsample(channels, 64, normalize=False),   #64*64*64
            *downsample(64, 128),            #32*32*128
            *downsample(128, 256),           #16*16*256
            *downsample(256, 512),           #8*8*512
            *downsample(512, 512),           #4*4*512
            nn.Conv2d(512, 4000, 1),         #4*4*4000  ?
            *upsample(4000, 512),            #8*8*512    
            *upsample(512, 256),             #16*16*256
            *upsample(256, 128),             #32*32*128
            *upsample(128, 128),             #64*64*128
            *upsample(128, 64),              #128*128*64
            nn.Conv2d(64, channels, 3, 1, 1),
            nn.Tanh()
        )
        '''
        self.down_size = img_size//32 
        num_filters = 512
        self.num_filters = num_filters
        down_dim = self.num_filters * (img_size//32) ** 2              # dim of feature: down_dim(2048)
        
        # encoder
        self.down = nn.Sequential(                        #128*128*3      
            *downsample(channels, 64, normalize=False),   #64*64*64       2
            *downsample(64, 64),                          #32*32*64       4
            *downsample(64, 128),                         #16*16*128      8
            *downsample(128, 256),                        #8*8*256        16
            *downsample(256, num_filters),                #4*4*512        32
        )
        
        #self.fc = nn.Sequential(
        #    nn.Linear(down_dim, 32),
        #    nn.BatchNorm1d(32, 0.8),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(32, down_dim),
        #    nn.BatchNorm1d(down_dim),
        #    nn.ReLU(inplace=True),
        #)
        # extract hide layer
        self.fc1 = nn.Linear(down_dim, num_filters)
        #self.fc2 = nn.MaxPool1d( 2 , 2 , 0)
        
        self.predict = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_filters, num_classes)
        )
        
        # decoder
        self.up = nn.Sequential(             #4*4*512
            nn.Conv2d(num_filters, 4000, 1), #4*4*4000
            *upsample(4000, 512),            #8*8*512
            *upsample(512, 256),             #16*16*256
            *upsample(256, 128),             #32*32*128
            *upsample(128, 64),              #64*64*64
            *upsample(64, 64),               #128*128*64
            nn.Conv2d(64, channels, 3, 1, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        out = self.down(x)        #4*4*512 
        features = self.fc1(out.view(out.size(0), -1)) #2048 -> 4*4*128 
        #features = self.fc2( fc1.view( fc1.size()[0] , -1 , 2  )).view( fc1.size()[0] , -1 ) 
        validity = self.predict(features)     #512 -> 4*4*512
        out = self.up(out)  #128*128*3 
        return validity, features, out  #validity, 

    
'''    
class FeaturePredict(nn.Module):
    def __init__(self ,  num_classes , global_feature_layer_dim = 8192 , dropout = 0.3):
        super(FeaturePredict,self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(global_feature_layer_dim , num_classes)
    def forward(self ,x ,use_dropout):
        if use_dropout:
            x = self.dropout(x)
        x = self.fc(x)
        return x
'''        
    
''' 
class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()
        
        img_size = 128
        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers
        
        self.down_size = img_size//8 
        down_dim = 32 * (img_size//8) ** 2               # dinm of feature: down_dim(8192)
        
        self.down = nn.Sequential(                       #128*128*3
            *downsample(channels, 64, normalize=False),   #64*64*64   2
            *downsample(64, 64),            #32*32*64                 4
            *downsample(64, 32),            #16*16*32                 8
        )
        self.fc = nn.Sequential(
            nn.Linear(down_dim, 32),
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Sequential(             #16*16*32      
            *upsample(down_dim, 64),         #32*32*64    
            *upsample(64, 64),               #64*64*64
            nn.Conv2d(64, channels, 3, 1, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        out = self.down(img)        #1*1*8192 
        features = self.fc(out.view(out.size(0), -1)) #8192 -> 16*16*32
        out = self.up(out.view(features.size(0), 32, self.down_size, self.down_size))  #128*128*3 
        return features, out    
'''    
       
class DiscriminatorGlobal(nn.Module):
    def __init__(self, channels=3):
        super(DiscriminatorGlobal, self).__init__()

        img_size = 128
        # downsampling
        self.down1 = nn.Sequential(nn.Conv2d(channels, 32, 3, 1, 1), nn.LeakyReLU(0.2),   #128
                                   ResidualBlock(32 , activation = nn.LeakyReLU()) )
        self.down2 = nn.Sequential(nn.Conv2d(32, 32, 4, 2, 1), nn.LeakyReLU(0.2),         #64
                                   ResidualBlock(32 , activation = nn.LeakyReLU()) )
        self.down3 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2),         #32
                                   ResidualBlock(64 , activation = nn.LeakyReLU()) )        
        '''
        # Fully-connected layers
        self.down_size = img_size//4 
        down_dim = 64 * (img_size//4) ** 2
        self.fc = nn.Sequential(
            nn.Linear(down_dim, 32),
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )
        '''
        # Upsampling
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), #64 
                                     nn.BatchNorm2d(32, 0.8),
                                     nn.ReLU())              
        self.up2 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1),       
                                 nn.BatchNorm2d(32, 0.8),
                                 nn.LeakyReLU(0.2),    
                                 ResidualBlock(32 , activation = nn.LeakyReLU()) )
        
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), #128
                                     nn.BatchNorm2d(32, 0.8),
                                     nn.ReLU()) 
        self.up1 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), 
                                 nn.BatchNorm2d(32, 0.8),
                                 nn.LeakyReLU(0.2),
                                 ResidualBlock(32 , activation = nn.LeakyReLU()) ) 
        
        self.output = nn.Conv2d(32, channels, 3, 1, 1)

    def forward(self, img):        #128*128*64
        down1 = self.down1(img)       
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        deconv2 = self.deconv2(down3) 
        up2 = self.up2(torch.cat([deconv2,down2],  1))
        deconv1 = self.deconv1(up2)
        up1 = self.up1(torch.cat([deconv1,down1],  1))
        output = self.output(up1)
        return output
    
class DiscriminatorLocal(nn.Module):
    def __init__(self, channels=3):
        super(DiscriminatorLocal, self).__init__()

        img_size = 64
        # downsampling
        self.down1 = nn.Sequential(nn.Conv2d(channels, 32, 3, 1, 1), nn.LeakyReLU(0.2),         #64
                                   ResidualBlock(32 , activation = nn.LeakyReLU()) )
        self.down2 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2),               #32
                                   ResidualBlock(64 , activation = nn.LeakyReLU()) )        

        # Upsampling
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),        #64 
                                     nn.BatchNorm2d(32, 0.8),
                                     nn.ReLU())              
        self.up1 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1),       
                                 nn.BatchNorm2d(32, 0.8),
                                 nn.LeakyReLU(0.2),    
                                 ResidualBlock(32 , activation = nn.LeakyReLU()) )
        self.output = nn.Conv2d(32, channels, 3, 1, 1)

    def forward(self, img):        #128*128*64
        down1 = self.down1(img)       
        down2 = self.down2(down1)
        deconv1 = self.deconv1(down2) 
        up1 = self.up1(torch.cat([deconv1,down1],  1))  
        output = self.output(up1)
        return output

class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()
        self.local_ = DiscriminatorLocal(channels)
        
        self.global_ = DiscriminatorGlobal(channels)
        self.fussion = nn.Conv2d(6, channels, 3, 1, 1)
        
    def forward(self, img):        
        local_left_up = self.local_(img[:,:,0:64,0:64])
        local_right_up = self.local_(img[:,:,0:64,64:128])
        local_left_down = self.local_(img[:,:,64:128,0:64])
        local_right_down = self.local_(img[:,:,64:128,64:128])
        
        up = torch.cat([local_left_up,local_right_up],  2)
        down = torch.cat([local_left_down,local_left_down],  2)    
        local_img = torch.cat([up,down],  3)
        global_img = self.global_(img)
        img = torch.cat([local_img,global_img],  1)
        output = self.fussion(img)
        return output  
        
        