from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt


output_dir = "phase_objects_train"
os.makedirs(output_dir, exist_ok=True)

input_folder = "train"
imgs = os.listdir(input_folder)

for x in range(len(imgs)):
    input_image_path=os.path.join(input_folder, imgs[x])
    image = Image.open(input_image_path).convert("L")
    image = np.array(image, dtype=np.float64)
    phase = (image / 255.0) * 0.314
    output_path = os.path.join(output_dir, imgs[x][:-5]+'.npy')
    np.save(output_path, phase)
    print(x)

output_dir = "phase_objects_test"
os.makedirs(output_dir, exist_ok=True)

input_folder = "test"
imgs = os.listdir(input_folder)

for x in range(len(imgs)):
    input_image_path=os.path.join(input_folder, imgs[x])
    image = Image.open(input_image_path).convert("L")
    image = np.array(image, dtype=np.float64)
    phase = (image / 255.0) * 0.314
    output_path = os.path.join(output_dir, imgs[x][:-5]+'.npy')
    np.save(output_path, phase)
    print(x)
    
output_dir = "phase_objects_val"
os.makedirs(output_dir, exist_ok=True)

input_folder = "val"
imgs = os.listdir(input_folder)

for x in range(len(imgs)):
    input_image_path=os.path.join(input_folder, imgs[x])
    image = Image.open(input_image_path).convert("L")
    image = np.array(image, dtype=np.float64)
    phase = (image / 255.0) * 0.314
    output_path = os.path.join(output_dir, imgs[x][:-5]+'.npy')
    np.save(output_path, phase)
    print(x)

    
vis = np.load('phase_objects_train/n01443537_0.npy')
plt.imshow(vis, cmap='gray')
plt.colorbar()
plt.show()

vis = np.load('phase_objects_test/test_0.npy')
plt.imshow(vis, cmap='gray')
plt.colorbar()
plt.show()

vis = np.load('phase_objects_val/val_0.npy')
plt.imshow(vis, cmap='gray')
plt.colorbar()
plt.show()