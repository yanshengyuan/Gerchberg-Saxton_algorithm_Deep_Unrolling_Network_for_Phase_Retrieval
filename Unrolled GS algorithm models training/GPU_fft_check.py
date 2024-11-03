import numpy as np
import torch

x = torch.rand((4, 4), dtype=torch.float32)
y = torch.rand((4, 4), dtype=torch.float32)

# Move the tensor to the GPU
x = x.cuda()
y = y.cuda()

# Perform the 2D FFT on the GPU
c=x*torch.exp(y * 1j)
fft_torch = torch.fft.fft2(c)

x = x.cpu()
y=y.cpu()
c=x*np.exp(y * 1j)
fft_np = np.fft.fft2(c)

#print(fft_torch)
#print(fft_np)

x = torch.rand((3, 4, 4), dtype=torch.float32)
y = torch.rand((3, 4, 4), dtype=torch.float32)
c=x*torch.exp(y * 1j)
fft_torch = torch.fft.fft2(c)
print(fft_torch)

x_0 = x[0]
y_0 = y[0]
c=x_0*torch.exp(y_0 * 1j)
fft_torch = torch.fft.fft2(c)
print(fft_torch)