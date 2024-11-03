import os
import numpy as np
from tensorflow.keras.datasets import mnist
from PIL import Image

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_dir = './mnist_train_images'
test_dir = './mnist_test_images'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for i, image in enumerate(train_images):
    img = Image.fromarray(image)
    img = img.resize((64, 64))
    img.save(os.path.join(train_dir, f'{i}_{train_labels[i]}.png'))

for i, image in enumerate(test_images):
    img = Image.fromarray(image)
    img = img.resize((64, 64))
    img.save(os.path.join(test_dir, f'{i}_{test_labels[i]}.png'))