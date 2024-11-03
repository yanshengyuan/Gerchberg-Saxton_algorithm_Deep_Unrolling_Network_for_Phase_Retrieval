import torch
import torch.nn as nn
import torch.nn.functional as f
from einops import rearrange
from LightPipes import *
import numpy as np
import matplotlib.pyplot as plt

wavelength=1000.0*nm
size=5*mm
N=64
z=0.6*m

class FFTLayer(nn.Module):
    def __init__(self, intensity_near):
        super(FFTLayer, self).__init__()
        self.conv_update = Phase_UpdateLayer()
        self.intens_near = intensity_near
        
    def forward(self, phase, intens_far):
        near_intens = self.intens_near
        near_phase = phase
        near_phase = self.conv_update(near_phase)
        near_field = near_intens*torch.exp(near_phase * 1j)
        
        far_field = torch.fft.fft2(near_field)
        
        far_intens = intens_far
        far_phase = torch.angle(far_field)
        far_phase = self.conv_update(far_phase)
        far_field = far_intens*torch.exp(far_phase * 1j)
        
        near_field = torch.fft.ifft2(far_field) 
        near_phase = torch.angle(near_field)
        
        return near_phase
    
class FresnelLayer_single(nn.Module):
    def __init__(self, intensity_near):
        super(FresnelLayer_single, self).__init__()
        self.intens_near = intensity_near
        self.F = Begin(size, wavelength, N)
        self.unet_update = UNet_res()
        
    def forward(self, phase, intens_far):
        with torch.no_grad():
            phase = phase.cpu().squeeze()
            F_=SubIntensity(self.intens_near, self.F)
            F_=SubPhase(phase, F_)
            F_=Forvard(F_, z, usepyFFTW=True)
            F_=SubIntensity(intens_far, F_)
            F_=Forvard(F_, -z, usepyFFTW=True)
            phase_upd = Phase(F_)
        phase_upd = torch.from_numpy(phase_upd).cuda().float()
        phase_upd = phase_upd.unsqueeze(0)
        phase_upd = self.unet_update(phase_upd)
        
        return phase_upd
    
class FresnelLayer(nn.Module):
    def __init__(self, intensity_near):
        super(FresnelLayer, self).__init__()
        self.intens_near = intensity_near
        self.F = Begin(size, wavelength, N)
        self.conv_update_far = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(inplace=True),
        )
        self.unet_update_near = UNet_res()
        
    def forward(self, phase, intens_far):
        with torch.no_grad():
            phase = phase.cpu().squeeze()
            F_ = SubIntensity(self.intens_near, self.F)
            F_ = SubPhase(phase, F_)
            F_ = Forvard(F_, z, usepyFFTW=True)
            phase_far_upd = Phase(F_)
        phase_far_upd = torch.from_numpy(phase_far_upd).cuda().float()
        phase_far_upd = phase_far_upd.unsqueeze(0)
        phase_far = self.conv_update_far(phase_far_upd)
        with torch.no_grad():
            phase_far = phase_far.cpu().squeeze()
            F_ = SubPhase(phase_far, F_)
            F_ = SubIntensity(intens_far, F_)
            F_ = Forvard(F_, -z, usepyFFTW=True)
            phase_near_upd = Phase(F_)
        phase_near_upd = torch.from_numpy(phase_near_upd).cuda().float()
        phase_near_upd = phase_near_upd.unsqueeze(0)
        phase_upd = self.unet_update_near(phase_near_upd)
        
        return phase_upd
    
class GSNet_FFT(nn.Module):
    def __init__(self, intensity_near, num_layers=30):
        super(GSNet_FFT, self).__init__()
        self.intens_near = intensity_near
        self.layers = nn.ModuleList()
        self.input_layer = FFTLayer(self.intens_near)
        
        for _ in range(num_layers):
            layer = FFTLayer(self.intens_near)
            self.layers.append(layer)
        
    def forward(self, phase_init, intens_far):
        phase = self.input_layer(phase_init, intens_far)
        for layer in self.layers:
            phase = layer(phase, intens_far)
        
        return phase
 
class GSNet_Fresnel(nn.Module):
    def __init__(self, intensity_near, num_layers=10, val=False):
        super(GSNet_Fresnel, self).__init__()
        self.intens_near = intensity_near
        self.layers = nn.ModuleList()
        self.F_fin = Begin(size, wavelength, N)
        
        for _ in range(num_layers):
            layer = FresnelLayer(self.intens_near)
            self.layers.append(layer)
            
        self.val = val
        
    def forward(self, intens_far):
        F_init = Begin(size,wavelength,N)
        phase = Phase(F_init)
        phase = torch.from_numpy(phase).cuda()
        for layer in self.layers:
            phase = layer(phase, intens_far)
        phase = phase.unsqueeze(1)
        if self.val==False:
            return phase
        with torch.no_grad():
            phase_fin = phase.cpu().squeeze()
            F_fin = SubIntensity(self.intens_near, self.F_fin)
            F_fin = SubPhase(phase_fin, F_fin)
            F_fin = Forvard(F_fin, z, usepyFFTW=True)
            I = Intensity(F_fin, flag=2)
        I=torch.tensor(I, dtype=torch.float32, requires_grad=True).cuda()
        return phase, I
    
class GSNet_Fresnel_selfsupervised(nn.Module):
    def __init__(self, intensity_near, num_layers=10):
        super(GSNet_Fresnel_selfsupervised, self).__init__()
        self.intens_near = intensity_near
        self.layers = nn.ModuleList()
        self.F_fin = Begin(size, wavelength, N)
        
        for _ in range(num_layers):
            layer = FresnelLayer(self.intens_near)
            self.layers.append(layer)
        
    def forward(self, intens_far):
        dummy_output=0
        F_init = Begin(size,wavelength,N)
        phase = Phase(F_init)
        phase = torch.from_numpy(phase).cuda()
        for layer in self.layers:
            phase = layer(phase, intens_far)
        phase = phase.squeeze()
        return phase, dummy_output
        phase_cpu = phase.clone().detach().to('cpu')
        with torch.no_grad():
            phase_fin = phase_cpu
            F_fin = SubIntensity(self.intens_near, self.F_fin)
            F_fin = SubPhase(phase_fin, F_fin)
            F_fin = Forvard(F_fin, z, usepyFFTW=True)
            I = Intensity(F_fin, flag=2)
        #plt.imsave('Phi.png', phase.squeeze().detach().cpu().numpy(), cmap='gray')
        for i in range(len(I)):
            for j in range(len(I[i])):
                phase[i][j] = I[i][j]
        #return dummy_output, phase

#########################################
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out


class UNet(nn.Module):
    def __init__(self, block=ConvBlock,dim=32):
        super(UNet, self).__init__()

        self.ConvBlock1 = ConvBlock(1, dim, strides=1)
        self.pool1 = nn.Conv2d(dim,dim,kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = block(dim, dim*2, strides=1)
        self.pool2 = nn.Conv2d(dim*2,dim*2,kernel_size=4, stride=2, padding=1)
       
        self.ConvBlock3 = block(dim*2, dim*4, strides=1)
        self.pool3 = nn.Conv2d(dim*4,dim*4,kernel_size=4, stride=2, padding=1)
       
        self.ConvBlock4 = block(dim*4, dim*8, strides=1)
        self.pool4 = nn.Conv2d(dim*8, dim*8,kernel_size=4, stride=2, padding=1)

        self.ConvBlock5 = block(dim*8, dim*16, strides=1)

        self.upv6 = nn.ConvTranspose2d(dim*16, dim*8, 2, stride=2)
        self.ConvBlock6 = block(dim*8, dim*8, strides=1)

        self.upv7 = nn.ConvTranspose2d(dim*8, dim*4, 2, stride=2)
        self.ConvBlock7 = block(dim*4, dim*4, strides=1)

        self.upv8 = nn.ConvTranspose2d(dim*4, dim*2, 2, stride=2)
        self.ConvBlock8 = block(dim*2, dim*2, strides=1)

        self.upv9 = nn.ConvTranspose2d(dim*2, dim, 2, stride=2)
        self.ConvBlock9 = block(dim, dim, strides=1)

        self.conv10 = nn.Conv2d(dim, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1 = self.ConvBlock1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.ConvBlock4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.ConvBlock5(pool4)

        up6 = self.upv6(conv5)
        #up6 = torch.cat([up6, conv4], 1)
        conv6 = self.ConvBlock6(up6)

        up7 = self.upv7(conv6)
        #up7 = torch.cat([up7, conv3], 1)
        conv7 = self.ConvBlock7(up7)

        up8 = self.upv8(conv7)
        #up8 = torch.cat([up8, conv2], 1)
        conv8 = self.ConvBlock8(up8)

        up9 = self.upv9(conv8)
        #up9 = torch.cat([up9, conv1], 1)
        conv9 = self.ConvBlock9(up9)

        conv10 = self.conv10(conv9)
        out = conv10
        
        #out=out.squeeze()

        return out
    
class UNet_res(nn.Module):
    def __init__(self, block=ConvBlock,dim=32):
        super(UNet_res, self).__init__()

        self.dim = dim
        self.ConvBlock1 = ConvBlock(1, dim, strides=1)
        self.pool1 = nn.Conv2d(dim,dim,kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = block(dim, dim*2, strides=1)
        self.pool2 = nn.Conv2d(dim*2,dim*2,kernel_size=4, stride=2, padding=1)
       
        self.ConvBlock3 = block(dim*2, dim*4, strides=1)
        self.pool3 = nn.Conv2d(dim*4,dim*4,kernel_size=4, stride=2, padding=1)
       
        self.ConvBlock4 = block(dim*4, dim*8, strides=1)
        self.pool4 = nn.Conv2d(dim*8, dim*8,kernel_size=4, stride=2, padding=1)

        self.ConvBlock5 = block(dim*8, dim*16, strides=1)

        self.upv6 = nn.ConvTranspose2d(dim*16, dim*8, 2, stride=2)
        self.ConvBlock6 = block(dim*16, dim*8, strides=1)

        self.upv7 = nn.ConvTranspose2d(dim*8, dim*4, 2, stride=2)
        self.ConvBlock7 = block(dim*8, dim*4, strides=1)

        self.upv8 = nn.ConvTranspose2d(dim*4, dim*2, 2, stride=2)
        self.ConvBlock8 = block(dim*4, dim*2, strides=1)

        self.upv9 = nn.ConvTranspose2d(dim*2, dim, 2, stride=2)
        self.ConvBlock9 = block(dim*2, dim, strides=1)

        self.conv10 = nn.Conv2d(dim, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1 = self.ConvBlock1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.ConvBlock4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.ConvBlock5(pool4)

        up6 = self.upv6(conv5)
        #print(up6.shape)
        #print(conv4.shape)
        up6 = torch.cat([up6, conv4], 0)
        conv6 = self.ConvBlock6(up6)

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 0)
        conv7 = self.ConvBlock7(up7)

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 0)
        conv8 = self.ConvBlock8(up8)

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 0)
        conv9 = self.ConvBlock9(up9)

        conv10 = self.conv10(conv9)
        out = x + conv10

        return out
    
class UNet_selfsupervised(nn.Module):
    def __init__(self, intensity_near, block=ConvBlock,dim=32):
        super(UNet_selfsupervised, self).__init__()

        self.ConvBlock1 = ConvBlock(1, dim, strides=1)
        self.pool1 = nn.Conv2d(dim,dim,kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = block(dim, dim*2, strides=1)
        self.pool2 = nn.Conv2d(dim*2,dim*2,kernel_size=4, stride=2, padding=1)
       
        self.ConvBlock3 = block(dim*2, dim*4, strides=1)
        self.pool3 = nn.Conv2d(dim*4,dim*4,kernel_size=4, stride=2, padding=1)
       
        self.ConvBlock4 = block(dim*4, dim*8, strides=1)
        self.pool4 = nn.Conv2d(dim*8, dim*8,kernel_size=4, stride=2, padding=1)

        self.ConvBlock5 = block(dim*8, dim*16, strides=1)

        self.upv6 = nn.ConvTranspose2d(dim*16, dim*8, 2, stride=2)
        self.ConvBlock6 = block(dim*8, dim*8, strides=1)

        self.upv7 = nn.ConvTranspose2d(dim*8, dim*4, 2, stride=2)
        self.ConvBlock7 = block(dim*4, dim*4, strides=1)

        self.upv8 = nn.ConvTranspose2d(dim*4, dim*2, 2, stride=2)
        self.ConvBlock8 = block(dim*2, dim*2, strides=1)

        self.upv9 = nn.ConvTranspose2d(dim*2, dim, 2, stride=2)
        self.ConvBlock9 = block(dim, dim, strides=1)

        self.conv10 = nn.Conv2d(dim, 1, kernel_size=3, stride=1, padding=1)
        
        self.F_fin = Begin(size, wavelength, N)
        self.intens_near = intensity_near

    def forward(self, x):
        conv1 = self.ConvBlock1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.ConvBlock4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.ConvBlock5(pool4)

        up6 = self.upv6(conv5)
        #up6 = torch.cat([up6, conv4], 1)
        conv6 = self.ConvBlock6(up6)

        up7 = self.upv7(conv6)
        #up7 = torch.cat([up7, conv3], 1)
        conv7 = self.ConvBlock7(up7)

        up8 = self.upv8(conv7)
        #up8 = torch.cat([up8, conv2], 1)
        conv8 = self.ConvBlock8(up8)

        up9 = self.upv9(conv8)
        #up9 = torch.cat([up9, conv1], 1)
        conv9 = self.ConvBlock9(up9)

        conv10 = self.conv10(conv9)
        phase = conv10
        
        phase = phase.squeeze()
        #return phase
        phase_cpu = phase.clone().detach().to('cpu')
        with torch.no_grad():
            phase_fin = phase_cpu
            F_fin = SubIntensity(self.intens_near, self.F_fin)
            F_fin = SubPhase(phase_fin, F_fin)
            F_fin = Forvard(F_fin, z, usepyFFTW=True)
            I = Intensity(F_fin, flag=2)
        #plt.imsave('Phi.png', phase.squeeze().detach().cpu().numpy(), cmap='gray')
        for i in range(len(I)):
            for j in range(len(I[i])):
                phase[i][j] = I[i][j]

        return phase
    
class UNet_original(nn.Module):
    def __init__(self, block=ConvBlock,dim=32):
        super(UNet_original, self).__init__()

        self.dim = dim
        self.ConvBlock1 = ConvBlock(1, dim, strides=1)
        self.pool1 = nn.Conv2d(dim,dim,kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = block(dim, dim*2, strides=1)
        self.pool2 = nn.Conv2d(dim*2,dim*2,kernel_size=4, stride=2, padding=1)
       
        self.ConvBlock3 = block(dim*2, dim*4, strides=1)
        self.pool3 = nn.Conv2d(dim*4,dim*4,kernel_size=4, stride=2, padding=1)
       
        self.ConvBlock4 = block(dim*4, dim*8, strides=1)
        self.pool4 = nn.Conv2d(dim*8, dim*8,kernel_size=4, stride=2, padding=1)

        self.ConvBlock5 = block(dim*8, dim*16, strides=1)

        self.upv6 = nn.ConvTranspose2d(dim*16, dim*8, 2, stride=2)
        self.ConvBlock6 = block(dim*16, dim*8, strides=1)

        self.upv7 = nn.ConvTranspose2d(dim*8, dim*4, 2, stride=2)
        self.ConvBlock7 = block(dim*8, dim*4, strides=1)

        self.upv8 = nn.ConvTranspose2d(dim*4, dim*2, 2, stride=2)
        self.ConvBlock8 = block(dim*4, dim*2, strides=1)

        self.upv9 = nn.ConvTranspose2d(dim*2, dim, 2, stride=2)
        self.ConvBlock9 = block(dim*2, dim, strides=1)

        self.conv10 = nn.Conv2d(dim, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1 = self.ConvBlock1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.ConvBlock4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.ConvBlock5(pool4)

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.ConvBlock6(up6)

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.ConvBlock7(up7)

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.ConvBlock8(up8)

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.ConvBlock9(up9)

        conv10 = self.conv10(conv9)
        out = x + conv10

        return out