#'''
from PIL import Image
import os
import numpy as np

output_dir = "test"
os.makedirs(output_dir, exist_ok=True)

input_image_folder = "test"
imgs = os.listdir(input_image_folder)

for x in range(len(imgs)):
    input_image_path=os.path.join(input_image_folder, imgs[x])
    image = Image.open(input_image_path).convert("L")
    image = image.resize((256, 256))
    image = image-np.min(image)
    image = (image/np.max(image))*255
    savepath=os.path.join(output_dir, imgs[x])
    image.save(savepath, "JPEG")
    print(x)
#'''

'''
from PIL import Image
import os

output_dir = "train"
os.makedirs(output_dir, exist_ok=True)

input_folder = "../tiny-imagenet/train"
classes = os.listdir(input_folder)

for i in range(len(classes)):
    class_path=os.path.join(input_folder, classes[i], "images")
    imgs=os.listdir(class_path)
    for j in range(len(imgs)):
        img_path=os.path.join(class_path, imgs[j])
        image = Image.open(img_path).convert("L")
        savepath=os.path.join(output_dir, imgs[j])
        image.save(savepath)
    print(i)
'''