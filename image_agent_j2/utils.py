import numpy as np
#import pystk
import torch

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from . import dense_transforms

RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15
DATASET_PATH = 'drive_data'
LENGTH_SCALE = 129
WH2 = np.array([400, 300])/2 # rescale factor of the image size
GOAL_H = 10.
GOAL_M = 5.
FRAME_NUM = 100*4 # frame * num * 4 (4 players)

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

def _in_view(x, proj, view):
        p = proj @ view @ np.array(list(x) + [1])
        return p[-1]

def obj_vec(kart, target, front):
    kartvec = np.array([front[0]-kart[0], front[2]- kart[2]])
    targetvec = target[::2]-kart[::2]
    dist = np.linalg.norm(targetvec)
    cos_angle = (kartvec @ targetvec)/dist/np.linalg.norm(kartvec)
    #return torch.tensor(dist), torch.tensor(cos_angle)
    return dist, cos_angle

def pointer(kart, target, front):
    kartvec = np.array([front[0]-kart[0], front[2]- kart[2]])
    targetvec = target[::2]-kart[::2]
    pointer = targetvec - kartvec
    return pointer/LENGTH_SCALE

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

            #"""
            proj_mat3 = np.array(datadict['team2_state'][0]['camera']['projection'])
            view_mat3 = np.array(datadict['team2_state'][0]['camera']['view'])
            loc3 = np.array(datadict['team2_state'][0]['kart']['location'])
            front3 = np.array(datadict['team2_state'][0]['kart']['front'])

            proj_mat4 = np.array(datadict['team2_state'][1]['camera']['projection'])
            view_mat4 = np.array(datadict['team2_state'][1]['camera']['view'])
            loc4 = np.array(datadict['team2_state'][1]['kart']['location'])
            front4 = np.array(datadict['team2_state'][1]['kart']['front'])

            #"""

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
            s1 = _to_image(puckloc, proj_mat1.T, view_mat1.T)
            s2 = _to_image(puckloc, proj_mat2.T, view_mat2.T)
            s3 = _to_image(puckloc, proj_mat3.T, view_mat3.T)
            s4 = _to_image(puckloc, proj_mat4.T, view_mat4.T)

            #print('-------------------- PUCK distance and angle---------------------------')
            #dist_to_obj(puckloc, proj_mat1.T, view_mat1.T)
            # team 1
            d1, a1 = obj_vec(loc1, puckloc, front1)
            d2, a2 = obj_vec(loc2, puckloc, front2)
            # team 2
            d3, a3 = obj_vec(loc3, puckloc, front3)
            d4, a4 = obj_vec(loc4, puckloc, front4)

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
            #d1g2, a1g2 = obj_vec(loc1, goal2, front1)
            #d2g2, a2g2 = obj_vec(loc2, goal2, front2)
            d3g2, a3g2 = obj_vec(loc3, goal2, front3)
            d4g2, a4g2 = obj_vec(loc4, goal2, front4)

            g1 = _to_image(goal1, proj_mat1.T, view_mat1.T)
            g2 = _to_image(goal1, proj_mat2.T, view_mat2.T)
            g3 = _to_image(goal2, proj_mat3.T, view_mat3.T)
            g4 = _to_image(goal2, proj_mat4.T, view_mat4.T)


            #print('----------------------PUCK Detections--------------------------------')
            puck_x1edge_1 = _to_image(puckloc + np.array([1.05, 0, 0]), proj_mat1.T, view_mat1.T)
            puck_x2edge_1 = _to_image(puckloc + np.array([-1.05, 0, 0]), proj_mat1.T, view_mat1.T)
            puck_x1edge_2 = _to_image(puckloc + np.array([1.05, 0, 0]), proj_mat2.T, view_mat2.T)
            puck_x2edge_2 = _to_image(puckloc + np.array([-1.05, 0, 0]), proj_mat2.T, view_mat2.T)
            puck_x1edge_3 = _to_image(puckloc + np.array([1.05, 0, 0]), proj_mat3.T, view_mat3.T)
            puck_x2edge_3 = _to_image(puckloc + np.array([-1.05, 0, 0]), proj_mat3.T, view_mat3.T)
            puck_x1edge_4 = _to_image(puckloc + np.array([1.05, 0, 0]), proj_mat4.T, view_mat4.T)
            puck_x2edge_4 = _to_image(puckloc + np.array([-1.05, 0, 0]), proj_mat4.T, view_mat4.T)


            puck_z1edge_1 = _to_image(puckloc + np.array([0, 0.5, 0]), proj_mat1.T, view_mat1.T)
            puck_z2edge_1 = _to_image(puckloc + np.array([0, -0.5, 0]), proj_mat1.T, view_mat1.T)
            puck_z1edge_2 = _to_image(puckloc + np.array([0, 0.5, 0]), proj_mat2.T, view_mat2.T)
            puck_z2edge_2 = _to_image(puckloc + np.array([0, -0.5, 0]), proj_mat2.T, view_mat2.T)
            puck_z1edge_3 = _to_image(puckloc + np.array([0, 0.5, 0]), proj_mat3.T, view_mat3.T)
            puck_z2edge_3 = _to_image(puckloc + np.array([0, -0.5, 0]), proj_mat3.T, view_mat3.T)
            puck_z1edge_4 = _to_image(puckloc + np.array([0, 0.5, 0]), proj_mat4.T, view_mat4.T)
            puck_z2edge_4 = _to_image(puckloc + np.array([0, -0.5, 0]), proj_mat4.T, view_mat4.T)

            # conversion to pixel locations:
            puck_x1edge_1 = WH2*(1+puck_x1edge_1)
            puck_x2edge_1 = WH2*(1+puck_x2edge_1)
            puck_z1edge_1 = WH2*(1+puck_z1edge_1)
            puck_z2edge_1 = WH2*(1+puck_z2edge_1)

            puck_x1edge_2 = WH2*(1+puck_x1edge_2)
            puck_x2edge_2 = WH2*(1+puck_x2edge_2)
            puck_z1edge_2 = WH2*(1+puck_z1edge_2)
            puck_z2edge_2 = WH2*(1+puck_z2edge_2)

            puck_x1edge_3 = WH2*(1+puck_x1edge_3)
            puck_x2edge_3 = WH2*(1+puck_x2edge_3)
            puck_z1edge_3 = WH2*(1+puck_z1edge_3)
            puck_z2edge_3 = WH2*(1+puck_z2edge_3)

            puck_x1edge_4 = WH2*(1+puck_x1edge_4)
            puck_x2edge_4 = WH2*(1+puck_x2edge_4)
            puck_z1edge_4 = WH2*(1+puck_z1edge_4)
            puck_z2edge_4 = WH2*(1+puck_z2edge_4)

            #print('----------------------GOAL Detections--------------------------------')
            #----------------------WIDTH-------------------------
            #---------Player 1-----------------------
            goal1_x1edge_1 = _to_image(np.array([-10, GOAL_M, 64.5]), proj_mat1.T, view_mat1.T)
            goal1_x2edge_1 = _to_image(np.array([10, GOAL_M, 64.5]), proj_mat1.T, view_mat1.T)
            goal2_x1edge_1 = _to_image(np.array([-10, GOAL_M, -64.5]), proj_mat1.T, view_mat1.T)
            goal2_x2edge_1 = _to_image(np.array([10, GOAL_M, -64.5]), proj_mat1.T, view_mat1.T)

            #---------Player 2-----------------------
            goal1_x1edge_2 = _to_image(np.array([-10, GOAL_M, 64.5]), proj_mat2.T, view_mat2.T)
            goal1_x2edge_2 = _to_image(np.array([10, GOAL_M, 64.5]), proj_mat2.T, view_mat2.T)
            goal2_x1edge_2 = _to_image(np.array([-10, GOAL_M, -64.5]), proj_mat2.T, view_mat2.T)
            goal2_x2edge_2 = _to_image(np.array([10, GOAL_M, -64.5]), proj_mat2.T, view_mat2.T)

            #---------Player 3-----------------------
            goal1_x1edge_3 = _to_image(np.array([-10, GOAL_M, 64.5]), proj_mat3.T, view_mat3.T)
            goal1_x2edge_3 = _to_image(np.array([10, GOAL_M, 64.5]), proj_mat3.T, view_mat3.T)
            goal2_x1edge_3 = _to_image(np.array([-10, GOAL_M, -64.5]), proj_mat3.T, view_mat3.T)
            goal2_x2edge_3 = _to_image(np.array([10, GOAL_M, -64.5]), proj_mat3.T, view_mat3.T)

            #---------Player 4-----------------------
            goal1_x1edge_4 = _to_image(np.array([-10, GOAL_M, 64.5]), proj_mat4.T, view_mat4.T)
            goal1_x2edge_4 = _to_image(np.array([10, GOAL_M, 64.5]), proj_mat4.T, view_mat4.T)
            goal2_x1edge_4 = _to_image(np.array([-10, GOAL_M, -64.5]), proj_mat4.T, view_mat4.T)
            goal2_x2edge_4 = _to_image(np.array([10, GOAL_M, -64.5]), proj_mat4.T, view_mat4.T)

            #----------------------HEIGHT-------------------------
            #---------Player 1-----------------------
            goal1_z1edge_1 = _to_image(np.array([0, 0.0, 64.5]), proj_mat1.T, view_mat1.T)
            goal1_z2edge_1 = _to_image(np.array([0, GOAL_H, 64.5]), proj_mat1.T, view_mat1.T)
            goal2_z1edge_1 = _to_image(np.array([0, 0.0, -64.5]), proj_mat1.T, view_mat1.T)
            goal2_z2edge_1 = _to_image(np.array([0, GOAL_H, -64.5]), proj_mat1.T, view_mat1.T)
            #---------Player 2-----------------------
            goal1_z1edge_2 = _to_image(np.array([0, 0.0, 64.5]), proj_mat2.T, view_mat2.T)
            goal1_z2edge_2 = _to_image(np.array([0, GOAL_H, 64.5]), proj_mat2.T, view_mat2.T)
            goal2_z1edge_2 = _to_image(np.array([0, 0.0, -64.5]), proj_mat2.T, view_mat2.T)
            goal2_z2edge_2 = _to_image(np.array([0, GOAL_H, -64.5]), proj_mat2.T, view_mat2.T)

            #---------Player 3-----------------------
            goal1_z1edge_3 = _to_image(np.array([0, 0.0, 64.5]), proj_mat3.T, view_mat3.T)
            goal1_z2edge_3 = _to_image(np.array([0, GOAL_H, 64.5]), proj_mat3.T, view_mat3.T)
            goal2_z1edge_3 = _to_image(np.array([0, 0.0, -64.5]), proj_mat3.T, view_mat3.T)
            goal2_z2edge_3 = _to_image(np.array([0, GOAL_H, -64.5]), proj_mat3.T, view_mat3.T)

            #---------Player 4-----------------------
            goal1_z1edge_4 = _to_image(np.array([0, 0.0, 64.5]), proj_mat4.T, view_mat4.T)
            goal1_z2edge_4 = _to_image(np.array([0, GOAL_H, 64.5]), proj_mat4.T, view_mat4.T)
            goal2_z1edge_4 = _to_image(np.array([0, 0.0, -64.5]), proj_mat4.T, view_mat4.T)
            goal2_z2edge_4 = _to_image(np.array([0, GOAL_H, -64.5]), proj_mat4.T, view_mat4.T)

            # conversion to pixel locations:
            #---------Player 1-----------------------
            goal1_x1edge_1 = WH2*(1+goal1_x1edge_1)
            goal1_x2edge_1 = WH2*(1+goal1_x2edge_1)
            goal1_z1edge_1 = WH2*(1+goal1_z1edge_1)
            goal1_z2edge_1 = WH2*(1+goal1_z2edge_1)
            goal2_x1edge_1 = WH2*(1+goal2_x1edge_1)
            goal2_x2edge_1 = WH2*(1+goal2_x2edge_1)
            goal2_z1edge_1 = WH2*(1+goal2_z1edge_1)
            goal2_z2edge_1 = WH2*(1+goal2_z2edge_1)
            #---------Player 2-----------------------
            goal1_x1edge_2 = WH2*(1+goal1_x1edge_2)
            goal1_x2edge_2 = WH2*(1+goal1_x2edge_2)
            goal1_z1edge_2 = WH2*(1+goal1_z1edge_2)
            goal1_z2edge_2 = WH2*(1+goal1_z2edge_2)
            goal2_x1edge_2 = WH2*(1+goal2_x1edge_2)
            goal2_x2edge_2 = WH2*(1+goal2_x2edge_2)
            goal2_z1edge_2 = WH2*(1+goal2_z1edge_2)
            goal2_z2edge_2 = WH2*(1+goal2_z2edge_2)
            #---------Player 3-----------------------
            goal1_x1edge_3 = WH2*(1+goal1_x1edge_3)
            goal1_x2edge_3 = WH2*(1+goal1_x2edge_3)
            goal1_z1edge_3 = WH2*(1+goal1_z1edge_3)
            goal1_z2edge_3 = WH2*(1+goal1_z2edge_3)
            goal2_x1edge_3 = WH2*(1+goal2_x1edge_3)
            goal2_x2edge_3 = WH2*(1+goal2_x2edge_3)
            goal2_z1edge_3 = WH2*(1+goal2_z1edge_3)
            goal2_z2edge_3 = WH2*(1+goal2_z2edge_3)

            #---------Player 3-----------------------
            goal1_x1edge_4 = WH2*(1+goal1_x1edge_4)
            goal1_x2edge_4 = WH2*(1+goal1_x2edge_4)
            goal1_z1edge_4 = WH2*(1+goal1_z1edge_4)
            goal1_z2edge_4 = WH2*(1+goal1_z2edge_4)
            goal2_x1edge_4 = WH2*(1+goal2_x1edge_4)
            goal2_x2edge_4 = WH2*(1+goal2_x2edge_4)
            goal2_z1edge_4 = WH2*(1+goal2_z1edge_4)
            goal2_z2edge_4 = WH2*(1+goal2_z2edge_4)

            

            #print('-------------------- IMAGE---------------------------------------')
            #"""
            t1img = datadict['team1_images']
            t1img = np.array(t1img) 
            t1img1 = t1img[0,:,:,:] 
            t1img2 = t1img[1,:,:,:]
            #team 2
            t2img = datadict['team2_images']
            t2img = np.array(t2img) 
            t2img1 = t2img[0,:,:,:] 
            t2img2 = t2img[1,:,:,:]
            
            
            #label1 = np.array([puck_x1edge_1[0], puck_z1edge_1[1], puck_x2edge_1[0] , puck_z2edge_1[1]])
            #label2 = np.array([puck_x1edge_2[0], puck_z1edge_2[1], puck_x2edge_2[0] , puck_z2edge_2[1]])
            #label3 = np.array([puck_x1edge_3[0], puck_z1edge_3[1], puck_x2edge_3[0] , puck_z2edge_3[1]])
            #label4 = np.array([puck_x1edge_4[0], puck_z1edge_4[1], puck_x2edge_4[0] , puck_z2edge_4[1]])
            #print('label 1: ', label1)
            label1 = [[puck_x1edge_1[0], puck_z1edge_1[1], puck_x2edge_1[0] , puck_z2edge_1[1]]]
            label2 = [[puck_x1edge_2[0], puck_z1edge_2[1], puck_x2edge_2[0] , puck_z2edge_2[1]]]
            label3 = [[puck_x1edge_3[0], puck_z1edge_3[1], puck_x2edge_3[0] , puck_z2edge_3[1]]]
            label4 = [[puck_x1edge_4[0], puck_z1edge_4[1], puck_x2edge_4[0] , puck_z2edge_4[1]]]
            #print('label 1: ', label1)

            goal1_1 = [[goal1_x1edge_1[0], goal1_z1edge_1[1], goal1_x2edge_1[0] , goal1_z2edge_1[1]]]
            goal2_1 = [[goal2_x1edge_1[0], goal2_z1edge_1[1], goal2_x2edge_1[0] , goal2_z2edge_1[1]]]

            goal1_2 = [[goal1_x1edge_2[0], goal1_z1edge_2[1], goal1_x2edge_2[0] , goal1_z2edge_2[1]]]
            goal2_2 = [[goal2_x1edge_2[0], goal2_z1edge_2[1], goal2_x2edge_2[0] , goal2_z2edge_2[1]]]

            goal1_3 = [[goal1_x1edge_3[0], goal1_z1edge_3[1], goal1_x2edge_3[0] , goal1_z2edge_3[1]]]
            goal2_3 = [[goal2_x1edge_3[0], goal2_z1edge_3[1], goal2_x2edge_3[0] , goal2_z2edge_3[1]]]

            goal1_4 = [[goal1_x1edge_4[0], goal1_z1edge_4[1], goal1_x2edge_4[0] , goal1_z2edge_4[1]]]
            goal2_4 = [[goal2_x1edge_4[0], goal2_z1edge_4[1], goal2_x2edge_4[0] , goal2_z2edge_4[1]]]

            #empty_label = [[None, None, None, None]]
            empty_label = [[]]

            # Check whether puck and goals are in the kart's view
            #---------Player 1-----------------------
            vp_1 = _in_view(puckloc, proj_mat1.T, view_mat1.T)
            vg1_1 = _in_view(goal1, proj_mat1.T, view_mat1.T)
            vg2_1 = _in_view(goal2, proj_mat1.T, view_mat1.T)
            if vp_1 < 0:
                label1 = empty_label
            if vg1_1 < 0:
                goal1_1 = empty_label
            if vg2_1 < 0:
                goal2_1 = empty_label
            #---------Player 2-----------------------
            vp_2 = _in_view(puckloc, proj_mat2.T, view_mat2.T)
            vg1_2 = _in_view(goal1, proj_mat2.T, view_mat2.T)
            vg2_2 = _in_view(goal2, proj_mat2.T, view_mat2.T)
            if vp_2 < 0:
                label2 = empty_label
            if vg1_2 < 0:
                goal1_2 = empty_label
            if vg2_2 < 0:
                goal2_2 = empty_label
            #---------Player 1-----------------------
            vp_3 = _in_view(puckloc, proj_mat3.T, view_mat3.T)
            vg1_3 = _in_view(goal1, proj_mat3.T, view_mat3.T)
            vg2_3 = _in_view(goal2, proj_mat3.T, view_mat3.T)
            if vp_3 < 0:
                label3 = empty_label
            if vg1_3 < 0:
                goal1_3 = empty_label
            if vg2_3 < 0:
                goal2_3 = empty_label
            #---------Player 1-----------------------
            vp_4 = _in_view(puckloc, proj_mat4.T, view_mat4.T)
            vg1_4 = _in_view(goal1, proj_mat4.T, view_mat4.T)
            vg2_4 = _in_view(goal2, proj_mat4.T, view_mat4.T)
            if vp_4 < 0:
                label4 = empty_label
            if vg1_4 < 0:
                goal1_4 = empty_label
            if vg2_4 < 0:
                goal2_4 = empty_label

            self.data.append((t1img1, label1, goal1_1, goal2_1))
            self.data.append((t1img2, label2, goal1_2, goal2_2))
            self.data.append((t2img1, label3, goal1_3, goal2_3))
            self.data.append((t2img2, label4, goal1_4, goal2_4))

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, puck, goal1, goal2 = self.data[idx]
        #img = self.transform(img)
        return (img, puck, goal1, goal2)   


def load_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=32):
    dataset = SuperTuxDataset(dataset_path, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)



if __name__ == '__main__':
    dataset = SuperTuxDataset('test.pkl')
    import torchvision.transforms.functional as F
    #from pylab import show, subplots
    import matplotlib.pyplot as plt
    #from matplotlib.patches import Circle
    import matplotlib.patches as patches
    import numpy as np
    #"""
    fig, axs = plt.subplots(2, 2, figsize=(16, 9))
    print('axes: ', axs)
    for i, ax in enumerate(axs.flat):
        im, puck, goal1, goal2 = dataset[FRAME_NUM+i]
        #print('image: ', im.shape)
        #print('label: ', puck)
        #hm, size = dense_transforms.detections_to_heatmap(label, im.shape[1:])
        #ax.imshow(F.to_pil_image(im), interpolation=None)
        ax.imshow(im)
        for k in puck:
            ax.add_patch(
                #patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], fc='none', ec='r', lw=2))
                patches.Rectangle((k[0], k[1]), k[2] - k[0], k[3] - k[1], fc='none', ec='g', lw=2))
            
        for k in goal1:
            ax.add_patch(
                #patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], fc='none', ec='r', lw=2))
                patches.Rectangle((k[0], k[1]), k[2] - k[0], k[3] - k[1], fc='none', ec='b', lw=2))
            
        for k in goal2:
            ax.add_patch(
                #patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], fc='none', ec='r', lw=2))
                patches.Rectangle((k[0], k[1]), k[2] - k[0], k[3] - k[1], fc='none', ec='r', lw=2))

        ax.axis('off')
    #"""
    fig, axs = plt.subplots(2, 2, figsize=(13, 6))
    for i, ax in enumerate(axs.flat):

        im, *dets = dataset[FRAME_NUM+i]
        hm, size = dense_transforms.detections_to_heatmap(dets, im.shape[0:2])
        #print('hm: ', torch.nonzero(hm))
        #print('size: ', torch.nonzero(size))
        #ax.imshow(F.to_pil_image(im), interpolation=None)
        ax.imshow(im)
        hm = hm.numpy().transpose([1, 2, 0])
        alpha = 0.25*hm.max(axis=2) + 0.75
        g = 1 - np.maximum(hm[:, :, 1], hm[:, :, 2])
        b = 1 - np.maximum(hm[:, :, 0], hm[:, :, 2])
        r = 1 - np.maximum(hm[:, :, 0], hm[:, :, 1])
        ax.imshow(np.stack((r, g, b, alpha), axis=2), interpolation=None)
        ax.axis('off')
    #fig.tight_layout()
    #"""

    plt.show()