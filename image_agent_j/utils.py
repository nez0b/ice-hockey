import numpy as np
#import pystk
import torch

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from . import dense_transforms

RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15
DATASET_PATH = 'drive_data'

def load_recording(recording):
    from pickle import load
    with open(recording, 'rb') as f:
        while True:
            try:
                yield load(f)
            except EOFError:
                break

def _to_image(x, proj, view):
        p = proj @ view @ np.array(list(x) + [1])
        return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

def obj_vec(kart, target, front):
    kartvec = np.array([front[0]-kart[0], front[2]- kart[2]])
    targetvec = target[::2]-kart[::2]
    dist = np.linalg.norm(targetvec)
    cos_angle = (kartvec @ targetvec)/dist/np.linalg.norm(kartvec)
    #return torch.tensor(dist), torch.tensor(cos_angle)
    return dist, cos_angle


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor()):
        from PIL import Image
        from glob import glob
        from os import path
        self.data = []

        for frame in load_recording(dataset_path):
    
            datadict = frame
            
            #print('-------------------- KART LOCATION on the frame image---------------------------')
            proj_mat1 = np.array(datadict['team1_state'][0]['camera']['projection'])
            view_mat1 = np.array(datadict['team1_state'][0]['camera']['view'])
            loc1 = np.array(datadict['team1_state'][0]['kart']['location'])
            front1 = np.array(datadict['team1_state'][0]['kart']['front'])

            proj_mat2 = np.array(datadict['team1_state'][1]['camera']['projection'])
            view_mat2 = np.array(datadict['team1_state'][1]['camera']['view'])
            loc2 = np.array(datadict['team1_state'][1]['kart']['location'])
            front2 = np.array(datadict['team1_state'][1]['kart']['front'])

            """
            proj_mat3 = np.array(datadict['team2_state'][0]['camera']['projection'])
            view_mat3 = np.array(datadict['team2_state'][0]['camera']['view'])
            loc3 = np.array(datadict['team2_state'][0]['kart']['location'])
            front3 = np.array(datadict['team2_state'][0]['kart']['front'])

            proj_mat4 = np.array(datadict['team2_state'][1]['camera']['projection'])
            view_mat4 = np.array(datadict['team2_state'][1]['camera']['view'])
            loc4 = np.array(datadict['team2_state'][1]['kart']['location'])
            front4 = np.array(datadict['team2_state'][1]['kart']['front'])

            """

            #print('proj mat shape: ', proj_mat1.shape)
            """
            # ----------- in kart1's point of view---------------------
            p11 = _to_image(loc1, proj_mat1.T, view_mat1.T)
            p21 =  _to_image(loc2, proj_mat1.T, view_mat1.T)
            p31 =  _to_image(loc3, proj_mat1.T, view_mat1.T)
            p41 = _to_image(loc4, proj_mat1.T, view_mat1.T)
            # ----------- in kart2's point of view---------------------
            p12 = _to_image(loc1, proj_mat2.T, view_mat2.T)
            p22 =  _to_image(loc2, proj_mat2.T, view_mat2.T)
            p32 =  _to_image(loc3, proj_mat2.T, view_mat2.T)
            p42 = _to_image(loc4, proj_mat2.T, view_mat2.T)
            # ----------- in kart3's point of view---------------------
            p13 = _to_image(loc1, proj_mat3.T, view_mat3.T)
            p23 =  _to_image(loc2, proj_mat3.T, view_mat3.T)
            p33 =  _to_image(loc3, proj_mat3.T, view_mat3.T)
            p43 = _to_image(loc4, proj_mat3.T, view_mat3.T)
            # ----------- in kart4's point of view---------------------
            p14 = _to_image(loc1, proj_mat4.T, view_mat4.T)
            p24 =  _to_image(loc2, proj_mat4.T, view_mat4.T)
            p34 =  _to_image(loc3, proj_mat4.T, view_mat4.T)
            p44 = _to_image(loc4, proj_mat4.T, view_mat4.T)
            """
           

            #print('-------------------- PUCK LOCATION on the frame image---------------------------')
            puckloc = np.array(datadict['soccer_state']['ball']['location'])
            #s1 = _to_image(puckloc, proj_mat1.T, view_mat1.T)
            #s2 = _to_image(puckloc, proj_mat2.T, view_mat2.T)
            #s3 = _to_image(puckloc, proj_mat3.T, view_mat3.T)
            #s4 = _to_image(puckloc, proj_mat4.T, view_mat4.T)

            #print('-------------------- PUCK distance and angle---------------------------')
            #dist_to_obj(puckloc, proj_mat1.T, view_mat1.T)
            d1, a1 = obj_vec(loc1, puckloc, front1)
            d2, a2 = obj_vec(loc2, puckloc, front2)
            #d3, a3 = obj_vec(loc3, puckloc, front3)
            #d4, a4 = obj_vec(loc4, puckloc, front4)

            #print('kart1, dist: ', d1, ' angle: ', np.arccos(a1))
            #print('kart2, dist: ', d2, ' angle: ', np.arccos(a2))
            #print('kart3, dist: ', d3, ' angle: ', np.arccos(a3))
            #print('kart4, dist: ', d4, ' angle: ', np.arccos(a4))

            #print('-------------------- GOAL distance and angle---------------------------')
            goal1 = np.array([0, 0.07, 64.5])
            goal2 = np.array([0, 0.07, -64.5])
            #dist_to_obj(puckloc, proj_mat1.T, view_mat1.T)
            d1g1, a1g1 = obj_vec(loc1, goal1, front1)
            d2g1, a2g1 = obj_vec(loc2, goal1, front2)
            d1g2, a1g2 = obj_vec(loc1, goal2, front1)
            d2g2, a2g2 = obj_vec(loc2, goal2, front2)

            #print('kart1, dist: ', d1g, ' angle: ', np.arccos(a1g))
            #print('kart2, dist: ', d2g, ' angle: ', np.arccos(a2g))
            #print('kart3, dist: ', d3g, ' angle: ', np.arccos(a3g))
            #print('kart4, dist: ', d4g, ' angle: ', np.arccos(a4g))


            #print('-------------------- IMAGE---------------------------------------')
            #"""
            t1img = datadict['team1_images']
            t1img = np.array(t1img) 
            t1img = t1img[0,:,:,:] 
            #t2img = datadict['team2_images']
            #t2img = np.array(t2img) 
            label = np.array([d1, a1, d1g1, a1g1])


            self.data.append((t1img, label))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        #img = self.transform(img)
        return (img, label)


def load_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=32):
    dataset = SuperTuxDataset(dataset_path, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)



if __name__ == '__main__':
    print('nothing in this file')