# Gerchberg-Saxton_algorithm_Deep_Unrolling_Network_for_Phase_Retrieval
A implementation of unrolling the iterative GS algorithm into an unrolled deep learning network. The GS iterative method interacts with deep learning process by intervals. The Fresnel Diffraction back-and-forth propagations in the network are implemented with LightPipes optical simulation on CPU. The deep learning layers in are Pytorch on GPU. Thus, during training or inference, the data is commuted between CPU and GPU constantly.

Results:

Metric: MAE(phase), phase~[0, 0.1*pi]
