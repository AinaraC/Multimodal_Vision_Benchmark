
from Camera_corruptions import ImageAddSunlight, ImageAddBrightness, ImageAddSunMono, ImageAddDarkness
from PIL import Image
import os
import numpy as np
import pickle
import torch

def load_image(image_path):
    image = Image.open(image_path)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image_np = np.array(image)
    image_rgb_uint8_np = image_np.astype(np.uint8)

    return image_rgb_uint8_np

def load_points(points_path):
    point_cloud = np.fromfile(points_path, dtype=np.float32)
    point_cloud = point_cloud.reshape(-1, 4)
    point_cloud_tensor = torch.from_numpy(point_cloud)

    return point_cloud_tensor


def save_image(result_path, sensor, corruption, severity, image_numpy_aug):
    image_aug = Image.fromarray(image_numpy_aug)
    path = os.path.join(result_path, sensor, corruption, f'severity{severity}.jpg')
    image_aug.save(path)

image_path = '/mmdetection3d/data/kitti/training/image_2/000000.png'
points_path = '/mmdetection3d/data/kitti/training/velodyne/000000.bin'
result_path = 'corruption_examples'
image_np = load_image(image_path)

darkness_sim = ImageAddDarkness(5, 2025)
img_aug4  = darkness_sim(image_np)
img_aug4 = Image.fromarray(img_aug4)
img_aug4.save('pruebadark5.jpg')

sunlight_sim = ImageAddSunlight(5, 2025)
img_aug =  sunlight_sim(image_np)
img_aug = Image.fromarray(img_aug)
img_aug.save('prueba5.jpg')

brightness_sim = ImageAddBrightness(2, 2025)
img_aug2 =  brightness_sim(image_np)
img_aug2 = Image.fromarray(img_aug2)
img_aug2.save('prueba6.jpg')

sunmono_sim  =  ImageAddSunMono(5)
img_aug3  = sunmono_sim(image_np)
img_aug3 = Image.fromarray(img_aug3)
img_aug3.save('prueba7.jpg')