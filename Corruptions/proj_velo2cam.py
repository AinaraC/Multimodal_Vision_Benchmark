# https://github.com/azureology/kitti-velo2cam
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

IM_CORRUPTIONS = ['snow', 'fog', 'rain', 'sunlight', 'darkness']
LIDAR_CORRUPTIONS = ['snow', 'fog', 'rain', 'sunlight']

def projection(binary, img, calib):

    # P2 (3 x 4) for left eye
    P2 = np.array([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3,4)
    R0_rect = np.array([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3,3)
    # Add a 1 in bottom-right, reshape to 4 x 4
    R0_rect = np.insert(R0_rect,3,values=[0,0,0],axis=0)
    R0_rect = np.insert(R0_rect,3,values=[0,0,0,1],axis=1)
    Tr_velo_to_cam = np.array([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3,4)
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam,3,values=[0,0,0,1],axis=0)
    # read raw data from binary
    scan = np.fromfile(binary, dtype=np.float32).reshape((-1,4))
    points = scan[:, 0:3] # lidar xyz (front, left, up)
    # TODO: use fov filter? 
    velo = np.insert(points,3,1,axis=1).T
    velo = np.delete(velo,np.where(velo[0,:]<0),axis=1)
    cam = P2.dot(R0_rect.dot(Tr_velo_to_cam.dot(velo)))
    cam = np.delete(cam,np.where(cam[2,:]<0),axis=1)
    # get u,v,z
    cam[:2] /= cam[2,:]
    # do projection staff
    plt.figure(figsize=(12,5),dpi=96,tight_layout=True)
    png = mpimg.imread(img)
    IMG_H,IMG_W,_ = png.shape
    # restrict canvas in range
    plt.axis([0,IMG_W,IMG_H,0])
    plt.imshow(png)
    # filter point out of canvas
    u,v,z = cam
    u_out = np.logical_or(u<0, u>IMG_W)
    v_out = np.logical_or(v<0, v>IMG_H)
    outlier = np.logical_or(u_out, v_out)
    cam = np.delete(cam,np.where(outlier),axis=1)
    # generate color map from depth
    u,v,z = cam
    plt.scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.5,s=2) 
    plt.axis('off')


def main():    
    with open(f'./kitti_ejemplos/000000.txt','r') as f:
        calib = f.readlines()
        
    projection('kitti_ejemplos/000000.bin', 'kitti_ejemplos/000000.png', calib)
    save_path = './corruption_examples/projection'
    os.makedirs(save_path, exist_ok=True)
    name = f'original.png'
    plt.savefig(os.path.join(save_path, name),bbox_inches='tight')

    
    for corruption in IM_CORRUPTIONS:
        for severity in range(1,6):
            img = f'./corruption_examples/camera/{corruption}/severity{severity}.jpg'
            if corruption in LIDAR_CORRUPTIONS:
                binary = f'./corruption_examples/lidar/{corruption}/severity{severity}.bin'
            else:
                binary = 'kitti_ejemplos/000000.bin'

            projection(binary, img, calib)
            
            dir_path = './corruption_examples/projection'
            save_path = os.path.join(dir_path, corruption)
            os.makedirs(save_path, exist_ok=True)
            name = f'severity{severity}.jpg'
            plt.savefig(os.path.join(dir_path, corruption, name),bbox_inches='tight')

if __name__ == '__main__':
    main()


