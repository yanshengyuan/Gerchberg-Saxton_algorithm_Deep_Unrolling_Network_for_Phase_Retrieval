#Phase-only imaging: GSNet_Fresnel

nohup python3 GSNet_Fresnel_training_phaseimaging.py --data ../MNIST --epochs 50 --batch_size 1 --gpu 0 --lr 0.0001 --step_size 3 --seed 12 --pth_name GS_UNetResConv_double_MNIST.pth.tar > GS_UNetResConv_double_MNIST.txt 2>&1 &

python3 GSNet_Fresnel_test_phaseimaging.py --gpu 0 --data ../MNIST -b 1 --pth_name GS_UNetResConv_double_MNIST.pth.tar

nohup python3 GSNet_Fresnel_training_phaseimaging.py --data ../IC_layout/sems --epochs 50 --batch_size 1 --gpu 0 --lr 0.0001 --step_size 3 --seed 12 --pth_name GS_UNetResConv_double_IClayout.pth.tar > GS_UNetResConv_double_IClayout.txt 2>&1 &

python3 GSNet_Fresnel_test_phaseimaging.py --gpu 0 --data ../IC_layout/sems -b 1 --pth_name GS_UNetResConv_double_IClayout.pth.tar

nohup python3 GSNet_Fresnel_training_phaseimaging.py --data ../tiny-imagenet-gray --epochs 50 --batch_size 1 --gpu 1 --lr 0.0001 --step_size 3 --seed 12 --pth_name GS_ConvResConvRes_double_imagenet.pth.tar --layers 10 > GS_ConvResConvRes_double_imagenet.txt 2>&1 &

python3 GSNet_Fresnel_test_phaseimaging.py --gpu 0 --data ../tiny-imagenet-gray -b 1 --pth_name GS_UNetResConv_double_imagenet.pth.tar --layers 10

#Phase-only imaging: UNet

nohup python3 UNet_training_phaseimaging.py --data ../MNIST --epochs 150 --batch_size 128 --gpu 0 --lr 0.0001 --step_size 5 --seed 12 --pth_name UNet_MNIST.pth.tar > UNet_MNIST.txt 2>&1 &

python3 UNet_test_phaseimaging.py --gpu 0 --data ../MNIST -b 1 --pth_name UNet_MNIST.pth.tar

nohup python3 UNet_training_phaseimaging.py --data ../IC_layout/sems --epochs 150 --batch_size 64 --gpu 0 --lr 0.0001 --step_size 5 --seed 12 --pth_name UNet_IClayout.pth.tar > UNet_IClayout.txt 2>&1 &

python3 UNet_test_phaseimaging.py --gpu 0 --data ../IC_layout/sems -b 1 --pth_name UNet_IClayout.pth.tar

nohup python3 UNet_training_phaseimaging.py --data ../tiny-imagenet-gray --epochs 150 --batch_size 128 --gpu 1 --lr 0.0001 --step_size 5 --seed 12 --pth_name UNet_imagenet.pth.tar > UNet_imagenet.txt 2>&1 &

python3 UNet_test_phaseimaging.py --gpu 0 --data ../tiny-imagenet-gray -b 1 --pth_name UNet_imagenet.pth.tar

#Wavefront sensing: GSNet_Fresnel

nohup python3 GSNet_Fresnel_training.py --data ../GaussZernike-npy-64 --epochs 150 --batch_size 1 --gpu 0 --lr 0.0001 --step_size 5 --seed 22 --pth_name GS_UNetResConv_double.pth.tar > GS_UNetResConv_double.txt 2>&1 &

python3 GSNet_Fresnel_test.py --gpu 0 --data ../GaussZernike-npy-64 -b 1 --pth_name GS_UNetResConv_double.pth.tar

#Wavefront sensing: UNet

nohup python3 UNet_training.py --data ../GaussZernike-npy-64 --epochs 150 --batch_size 128 --gpu 1 --lr 0.0001 --step_size 5 --seed 22 --pth_name UNet.pth.tar > UNet.txt 2>&1 &

python3 UNet_test.py --gpu 0 --data ../GaussZernike-npy-64 -b 1 --pth_name UNet.pth.tar

#Summary
Phase Imaging GSNet: UNetRes
Phase Imaging data-driven: UNet
Wavefront Sensing GSNet: UNetRes
Wavefront Sensing data-driven: UNet