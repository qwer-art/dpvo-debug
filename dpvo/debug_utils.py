import numpy as np
import torch
import os.path as osp
import os
import cv2
import pickle
debug_path = "/home/jerett/Project/DPVO/Debug"

def load_image(frame_time):
    image_dir = osp.join(debug_path,"image")
    image_path = osp.join(image_dir,f"{frame_time}.png")
    return cv2.imread(image_path,-1)

def save_image(frame_time,frame_image):
    ## 1.tensor->array
    image_array = frame_image.detach().cpu().numpy()
    image_final = np.transpose(image_array, (1, 2, 0))
    ## 2.mkdir
    image_dir = osp.join(debug_path,"image")
    if not osp.exists(image_dir):
        os.makedirs(image_dir,exist_ok=True)
    ## 3.save
    image_path = osp.join(image_dir,f"{frame_time}.png")
    cv2.imwrite(image_path,image_final)

def save_intrinsics(frame_time,intrinsics):
    ## 1.tensor->array
    intr_array = intrinsics.detach().cpu().numpy()
    ## 2.mkdir
    direct = osp.join(debug_path,"intrinsics")
    if not osp.exists(direct):
        os.makedirs(direct,exist_ok=True)
    ## 3.file_name
    filename = osp.join(direct,f"{frame_time}.txt")
    np.savetxt(filename, intr_array, fmt='%.6f') 

def load_intrinsics(frame_time):
    filename = osp.join(debug_path,"intrinsics",f"{frame_time}.txt")
    return np.loadtxt(filename)

keys = ["fmap_array","gmap_array","imap_array","patches_array","clr_array"]
def save_features(frame_time,pred_feature):
    ## 1.tensor->array
    fmap, gmap, imap, patches, _, clr  = pred_feature
    fmap_array = fmap.detach().cpu().numpy().squeeze()
    gmap_array = gmap.detach().cpu().numpy().squeeze()
    imap_array = imap.detach().cpu().numpy().squeeze()
    patches_array = patches.detach().cpu().numpy().squeeze()
    clr_array = clr.detach().cpu().numpy().squeeze()
    feature_dict = {
        "fmap_array": fmap_array,
        "gmap_array": gmap_array,
        "imap_array": imap_array,
        "patches_array": patches_array,
        "clr_array": clr_array,
    }
    ## 2.mkdir
    direct = osp.join(debug_path,"feature")
    if not osp.exists(direct):
        os.makedirs(direct,exist_ok=True)
    ## 3.file_name
    filename = osp.join(direct,f"{frame_time}.pkl")
    with open(filename, 'wb') as file:
        pickle.dump(feature_dict, file)

def load_features(frame_time):
    direct = osp.join(debug_path,"feature")
    filename = osp.join(direct,f"{frame_time}.pkl")
    with open(filename,"rb") as file:
        pkl_data = pickle.load(file)
    return pkl_data