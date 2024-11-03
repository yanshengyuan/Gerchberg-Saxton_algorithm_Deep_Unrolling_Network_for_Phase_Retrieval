Gerchberg-Saxton Algorithm Deep Unrolling Network for Phase Retrieval
A implementation of unrolling the iterative GS algorithm into an unrolled deep learning network. The GS iterative method interacts with deep learning process by intervals. The Fresnel Diffraction back-and-forth propagations in the network are implemented with LightPipes optical simulation on CPU. The deep learning layers in are Pytorch on GPU. Thus, during training or inference, the data is commuted between CPU and GPU constantly.




Results:

Phase Retrieval task: Near-field phase-only imaging (CDI)

Dataset: tiny ImageNet1K

Metric: SSIM (retrieved image, GT image), pix values~[0, 255]

Gerchberg-Saxton Algorithm Deep Unrolled Network: 0.79

Unet (end2end pure data-driven method): 0.55

Phase Retrieval task: High-power Gaussian laser beam aberration detection (wavefront sensing)

Dataset: InShaPe-inspired dataset

Metric: MAE(phase), phase~[-pi, +pi]

Gerchberg-Saxton Algorithm Deep Unrolled Network: 0.2453 rad

Unet (end2end pure data-driven method): 0.6923 rad

Original Gerchberg-Saxton Algorithm 2000 iterations: 2.7418 rad




To reproduce the results above, follow the steps below:

1, Generate the three training and testing datasets: Enter the folders "MNIST/", "tiny-imagenet-gray/", and "GaussZernike-npy-64/", corresponding to the datasets MNIST dataset, tiny ImageNet1K dataset, and InShaPe-inspired dataset, respectively. In each folder there are two simulation scripts named as "train.py" and "test.py". They are the optical simulation scripts to generate the training and testing data for this repository. Run these scripts to generate all the training and testing datasets.

2, Enter the folder "Unrolled GS algorithm models training/", the training and testing codes for the GS Algorithm Deep Unrolled Network and UNet pure data-driven model are in this folder. Please use the following commands to train and test on different datasets for different tasks

#Phase-only imaging: GSNet_Fresnel

python3 GSNet_Fresnel_training_phaseimaging.py --data ../MNIST --epochs 50 --batch_size 1 --gpu 0 --lr 0.0001 --step_size 3 --seed 12 --pth_name GS_UNetResConv_double_MNIST.pth.tar

python3 GSNet_Fresnel_test_phaseimaging.py --gpu 0 --data ../MNIST -b 1 --pth_name GS_UNetResConv_double_MNIST.pth.tar

python3 GSNet_Fresnel_training_phaseimaging.py --data ../IC_layout/sems --epochs 50 --batch_size 1 --gpu 0 --lr 0.0001 --step_size 3 --seed 12 --pth_name GS_UNetResConv_double_IClayout.pth.tar

python3 GSNet_Fresnel_test_phaseimaging.py --gpu 0 --data ../IC_layout/sems -b 1 --pth_name GS_UNetResConv_double_IClayout.pth.tar

python3 GSNet_Fresnel_training_phaseimaging.py --data ../tiny-imagenet-gray --epochs 50 --batch_size 1 --gpu 1 --lr 0.0001 --step_size 3 --seed 12 --pth_name GS_ConvResConvRes_double_imagenet.pth.tar --layers 10

python3 GSNet_Fresnel_test_phaseimaging.py --gpu 0 --data ../tiny-imagenet-gray -b 1 --pth_name GS_UNetResConv_double_imagenet.pth.tar --layers 10


#Phase-only imaging: UNet

python3 UNet_training_phaseimaging.py --data ../MNIST --epochs 150 --batch_size 128 --gpu 0 --lr 0.0001 --step_size 5 --seed 12 --pth_name UNet_MNIST.pth.tar

python3 UNet_test_phaseimaging.py --gpu 0 --data ../MNIST -b 1 --pth_name UNet_MNIST.pth.tar

python3 UNet_training_phaseimaging.py --data ../IC_layout/sems --epochs 150 --batch_size 64 --gpu 0 --lr 0.0001 --step_size 5 --seed 12 --pth_name UNet_IClayout.pth.tar

python3 UNet_test_phaseimaging.py --gpu 0 --data ../IC_layout/sems -b 1 --pth_name UNet_IClayout.pth.tar


python3 UNet_training_phaseimaging.py --data ../tiny-imagenet-gray --epochs 150 --batch_size 128 --gpu 1 --lr 0.0001 --step_size 5 --seed 12 --pth_name UNet_imagenet.pth.tar

python3 UNet_test_phaseimaging.py --gpu 0 --data ../tiny-imagenet-gray -b 1 --pth_name UNet_imagenet.pth.tar


#Wavefront sensing: GSNet_Fresnel

python3 GSNet_Fresnel_training.py --data ../GaussZernike-npy-64 --epochs 150 --batch_size 1 --gpu 0 --lr 0.0001 --step_size 5 --seed 22 --pth_name GS_UNetResConv_double.pth.tar

python3 GSNet_Fresnel_test.py --gpu 0 --data ../GaussZernike-npy-64 -b 1 --pth_name GS_UNetResConv_double.pth.tar


#Wavefront sensing: UNet

python3 UNet_training.py --data ../GaussZernike-npy-64 --epochs 150 --batch_size 128 --gpu 1 --lr 0.0001 --step_size 5 --seed 22 --pth_name UNet.pth.tar

python3 UNet_test.py --gpu 0 --data ../GaussZernike-npy-64 -b 1 --pth_name UNet.pth.tar


#Summary
Phase Imaging GSNet: UNet with residual connections

Phase Imaging data-driven: UNet without residual connections

Wavefront Sensing GSNet: UNet with residual connections

Wavefront Sensing data-driven: UNet without residual connections


3, To run the GS algorithm (implemented by me), please open the script GS_Fresnel.py and specify the path of the test dataset as the target of the phase retrieval algorithm. And then run this script

python3 GS_Fresnel.py
