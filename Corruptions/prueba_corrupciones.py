
from Camera_corruptions import ImageAddSnow,ImageAddFog,ImageAddRain,ImageAddDarkness, ImageAddSunlight
from LiDAR_corruptions import rain_sim, fog_sim, scene_glare_noise, snow_sim
from PIL import Image
import os
import numpy as np
import pickle
import torch
import os


def load_image(image_path):
    image = Image.open(image_path)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image_np = np.array(image)
    image_rgb_uint8_np = image_np.astype(np.uint8)

    return image_rgb_uint8_np


def save_image(result_path, sensor, corruption, severity, image_numpy_aug):
    image_aug = Image.fromarray(image_numpy_aug)
    path = os.path.join(result_path, sensor, corruption, f'severity{severity}.jpg')
    image_aug.save(path)

def save_points(result_path, sensor, corruption, severity, points_np):  
    save_path = os.path.join(result_path, sensor, corruption, f'severity{severity}.bin')
    points_np = points_np.flatten().astype(np.float32)
    points_np.tofile(save_path)
    
def load_points(points_path):
    points = np.fromfile(points_path, dtype=np.float32)
    points = points.reshape((-1, 4))
    return points





image_path = './kitti_ejemplos/000000.png'
points_path = './kitti_ejemplos/000000.bin'

seed = 2025

camera = False
lidar = True

dir_path = './corruption_examples'

if camera:
    image_np = load_image(image_path)
    sensor  = 'camera'

    os.makedirs(os.path.join(dir_path,sensor), exist_ok=True)


    # Snow
    corruption = 'snow'
    os.makedirs(os.path.join(dir_path,sensor, corruption), exist_ok=True)

    snow_sim1 = ImageAddSnow(1, seed)
    snow_sim2 = ImageAddSnow(2, seed)
    snow_sim3 = ImageAddSnow(3, seed)
    snow_sim4 = ImageAddSnow(4, seed)
    snow_sim5 = ImageAddSnow(5, seed)

    img_aug = snow_sim1(image_np)
    save_image(dir_path, sensor, corruption ,1, img_aug  )
    img_aug = snow_sim2(image_np)
    save_image(dir_path, sensor, corruption ,2, img_aug  )
    img_aug = snow_sim3(image_np)
    save_image(dir_path, sensor, corruption ,3, img_aug  )
    img_aug = snow_sim4(image_np)
    save_image(dir_path, sensor, corruption ,4, img_aug  )
    img_aug = snow_sim5(image_np)
    save_image(dir_path, sensor, corruption ,5, img_aug  )

    # Fog
    corruption = 'fog'
    os.makedirs(os.path.join(dir_path,sensor, corruption), exist_ok=True)
    
    fog_sim1 = ImageAddFog(1, seed)
    fog_sim2 = ImageAddFog(2, seed)
    fog_sim3 = ImageAddFog(3, seed)
    fog_sim4 = ImageAddFog(4, seed)
    fog_sim5 = ImageAddFog(5, seed)

    img_aug = fog_sim1(image_np)
    save_image(dir_path, sensor, corruption ,1, img_aug  )
    img_aug = fog_sim2(image_np)
    save_image(dir_path, sensor, corruption ,2, img_aug  )
    img_aug = fog_sim3(image_np) 
    save_image(dir_path, sensor, corruption ,3, img_aug  )
    img_aug = fog_sim4(image_np)
    save_image(dir_path, sensor, corruption ,4, img_aug  )
    img_aug = fog_sim5(image_np)
    save_image(dir_path, sensor, corruption ,5, img_aug  )

    # Rain
    corruption = 'rain'
    os.makedirs(os.path.join(dir_path,sensor, corruption), exist_ok=True)

    rain_sim1 = ImageAddRain(1, seed)
    rain_sim2 = ImageAddRain(2, seed)
    rain_sim3 = ImageAddRain(3, seed)
    rain_sim4 = ImageAddRain(4, seed)
    rain_sim5 = ImageAddRain(5, seed)

    img_aug = rain_sim1(image_np)    
    save_image(dir_path, sensor, corruption ,1, img_aug  )
    img_aug = rain_sim2(image_np)
    save_image(dir_path, sensor, corruption ,2, img_aug  )
    img_aug = rain_sim3(image_np)
    save_image(dir_path, sensor, corruption ,3, img_aug  )
    img_aug = rain_sim4(image_np)
    save_image(dir_path, sensor, corruption ,4, img_aug  )
    img_aug = rain_sim5(image_np)
    save_image(dir_path, sensor, corruption ,5, img_aug  )

    # Sun
    corruption = 'sunlight'
    os.makedirs(os.path.join(dir_path,sensor, corruption), exist_ok=True)
    sunlight_sim1 = ImageAddSunlight(1, seed)
    sunlight_sim2 = ImageAddSunlight(2, seed)
    sunlight_sim3 = ImageAddSunlight(3, seed)
    sunlight_sim4 = ImageAddSunlight(4, seed)
    sunlight_sim5 = ImageAddSunlight(5, seed)
    img_aug = sunlight_sim1(image_np)
    save_image(dir_path, sensor, corruption ,1, img_aug)
    img_aug = sunlight_sim2(image_np)
    save_image(dir_path, sensor, corruption ,2, img_aug)
    img_aug = sunlight_sim3(image_np)
    save_image(dir_path, sensor, corruption ,3, img_aug)
    img_aug = sunlight_sim4(image_np)
    save_image(dir_path, sensor, corruption ,4, img_aug)
    img_aug = sunlight_sim5(image_np)  
    save_image(dir_path, sensor, corruption ,5, img_aug)

    # Night
    corruption = 'darkness'
    os.makedirs(os.path.join(dir_path,sensor, corruption), exist_ok=True)
    # Arreglar tipos de datos dentro de funcion
    darkness_sim1 = ImageAddDarkness(1, seed)
    darkness_sim2 = ImageAddDarkness(2, seed)
    darkness_sim3 = ImageAddDarkness(3, seed)
    darkness_sim4 = ImageAddDarkness(4, seed)
    darkness_sim5 = ImageAddDarkness(5, seed)


    img_aug = darkness_sim1(image_np)
    save_image(dir_path, sensor, corruption ,1, img_aug )
    img_aug = darkness_sim2(image_np)
    save_image(dir_path, sensor, corruption ,2, img_aug )
    img_aug = darkness_sim3(image_np)
    save_image(dir_path, sensor, corruption ,3, img_aug )
    img_aug = darkness_sim4(image_np)
    save_image(dir_path, sensor, corruption ,4, img_aug )
    img_aug = darkness_sim5(image_np)
    save_image(dir_path, sensor, corruption ,5, img_aug )


if lidar:
    points_np = load_points(points_path)

    sensor = 'lidar'
    os.makedirs(os.path.join(dir_path,sensor), exist_ok=True)
    # Snow
    corruption = 'rain'
    lidar_aug = None
    os.makedirs(os.path.join(dir_path,sensor, corruption), exist_ok=True)
    lidar_aug = rain_sim(points_np, 1) 
    save_points(dir_path, sensor, corruption ,1, lidar_aug[:, :4] )
    lidar_aug = rain_sim(points_np, 2)
    save_points(dir_path, sensor, corruption ,2, lidar_aug[:, :4] )
    lidar_aug = rain_sim(points_np, 3)
    save_points(dir_path, sensor, corruption ,3, lidar_aug[:, :4] )
    lidar_aug = rain_sim(points_np, 4)
    save_points(dir_path, sensor, corruption ,4, lidar_aug[:, :4] )
    lidar_aug = rain_sim(points_np, 5)
    save_points(dir_path, sensor, corruption ,5, lidar_aug[:, :4] )
    
    corruption = 'snow'
    lidar_aug = None
    os.makedirs(os.path.join(dir_path,sensor, corruption), exist_ok=True)
    lidar_aug = snow_sim(points_np, 1)  
    save_points(dir_path, sensor, corruption ,1, lidar_aug[:, :4] )
    lidar_aug = snow_sim(points_np, 2)
    save_points(dir_path, sensor, corruption ,2, lidar_aug[:, :4] )
    lidar_aug = snow_sim(points_np, 3)
    save_points(dir_path, sensor, corruption ,3, lidar_aug[:, :4] )
    lidar_aug = snow_sim(points_np, 4) 
    save_points(dir_path, sensor, corruption ,4, lidar_aug[:, :4] )
    lidar_aug = snow_sim(points_np, 5)
    save_points(dir_path, sensor, corruption ,5, lidar_aug[:, :4] )
    
    corruption = 'fog'
    lidar_aug = None
    #https://github.com/MartinHahner/LiDAR_fog_sim lookup tables
    os.makedirs(os.path.join(dir_path,sensor, corruption), exist_ok=True)
    lidar_aug = fog_sim(points_np, 1)
    save_points(dir_path, sensor, corruption ,1, lidar_aug[:, :4]  )
    lidar_aug = fog_sim(points_np, 2) 
    save_points(dir_path, sensor, corruption ,2, lidar_aug[:, :4]  )
    lidar_aug = fog_sim(points_np, 3)
    save_points(dir_path, sensor, corruption ,3, lidar_aug[:, :4]  )
    lidar_aug = fog_sim(points_np, 4) 
    save_points(dir_path, sensor, corruption ,4, lidar_aug[:, :4]  )
    lidar_aug = fog_sim(points_np, 5)
    save_points(dir_path, sensor, corruption ,5, lidar_aug[:, :4]  )
    
    corruption = 'sunlight'
    lidar_aug = None
    os.makedirs(os.path.join(dir_path,sensor, corruption), exist_ok=True)
    lidar_aug = scene_glare_noise(points_np, 1) 
    save_points(dir_path, sensor, corruption ,1, lidar_aug[:, :4]  )
    lidar_aug = scene_glare_noise(points_np, 2)  
    save_points(dir_path, sensor, corruption ,2, lidar_aug[:, :4]  )
    lidar_aug = scene_glare_noise(points_np, 3) 
    save_points(dir_path, sensor, corruption ,3, lidar_aug[:, :4]  )
    lidar_aug = scene_glare_noise(points_np, 4) 
    save_points(dir_path, sensor, corruption ,4, lidar_aug[:, :4]  )
    lidar_aug = scene_glare_noise(points_np, 5)
    save_points(dir_path, sensor, corruption ,5, lidar_aug[:, :4]  )