#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM

Modified by 
@Author: An Tao, Pengliang Ji
@Contact: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn
@Time: 2021/7/20 7:49 PM

Modified by 
@Author: Oddy Virgantara Putra
@Contact: oddy@unida.gontor.ac.id
@Time: 2025/05/03 5:45 AM
"""


import os
import sys
import glob
import h5py
import numpy as np
import torch
import json
import cv2
from torch.utils.data import Dataset
from PointWOLF import PointWOLF

def download_modelnet40():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = BASE_DIR
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR,'..','..','data','modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ('modelnet40_ply_hdf5_2048', DATA_DIR))
        os.system('rm %s' % (zipfile))


def download_shapenetpart():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = BASE_DIR
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists('D:\\datasets\\shapenet_part_seg_hdf5_data'):
        www = 'https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ('hdf5_data', os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')))
        os.system('rm %s' % (zipfile))


def download_S3DIS():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = BASE_DIR
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists("D:\\datasets\\indoor3d_sem_seg_hdf5_data"):
        www = 'https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ('indoor3d_sem_seg_hdf5_data', DATA_DIR))
        os.system('rm %s' % (zipfile))
    if not os.path.exists("D:\datasets\Stanford3dDataset_v1.2_Aligned_Version"):
        if not os.path.exists(os.path.join(DATA_DIR,'..','..','data', 'Stanford3dDataset_v1.2_Aligned_Version.zip')):
            print('Please download Stanford3dDataset_v1.2_Aligned_Version.zip \
                from https://goo.gl/forms/4SoGp4KtH1jfRqEj2 and place it under data/')
            sys.exit(0)
        else:
            zippath = os.path.join(DATA_DIR, '..','..','data','Stanford3dDataset_v1.2_Aligned_Version.zip')
            os.system('unzip %s' % (zippath))
            os.system('mv %s %s' % ('Stanford3dDataset_v1.2_Aligned_Version', DATA_DIR))
            os.system('rm %s' % (zippath))
    # print("***************")
    

def load_data_cls(partition):
    # download_modelnet40()
    BASE_DIR = '/home/virgantara/PythonProjects/DualGraphPoint'
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def load_data_partseg(partition):
    download_shapenetpart()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = BASE_DIR
    all_data = []
    all_label = []
    all_seg = []
    if partition == 'trainval':
        file = glob.glob(os.path.join('D:\\datasets\\shapenet_part_seg_hdf5_data', '*train*.h5')) \
               + glob.glob(os.path.join('D:\\datasets\\shapenet_part_seg_hdf5_data', '*val*.h5'))
    else:
        file = glob.glob(os.path.join('D:\\datasets\\shapenet_part_seg_hdf5_data', '*%s*.h5'%partition))
    for h5_name in file:
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg


def prepare_test_data_semseg():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = BASE_DIR
    print("+++++++++++++++++++")
    if not os.path.exists('D:\\datasets\\stanford_indoor3d'):
        print("**************")
        os.system('python prepare_data\\collect_indoor3d_data.py')
    if not os.path.exists('D:\\datasets\\indoor3d_sem_seg_hdf5_data_test'):
        os.system('python prepare_data\\gen_indoor3d_h5.py')


def load_data_semseg(partition, test_area):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = BASE_DIR
    # print("******************")
    # download_S3DIS()
    # print("**************")
    prepare_test_data_semseg()
    if partition == 'train':
        data_dir = "D:\\datasets\\indoor3d_sem_seg_hdf5_data"
    else:
        data_dir = "D:\\datasets\\indoor3d_sem_seg_hdf5_data_test"
    with open(os.path.join(data_dir, "all_files.txt")) as f:
        all_files = [line.rstrip() for line in f]
    with open(os.path.join(data_dir, "room_filelist.txt")) as f:
        room_filelist = [line.rstrip() for line in f]
        
    data_batchlist, label_batchlist = [], []
    for f in all_files:
        file = h5py.File(os.path.join("D:\\datasets",f), 'r')
        data = file["data"][:]
        label = file["label"][:]
        data_batchlist.append(data)
        label_batchlist.append(label)
    data_batches = np.concatenate(data_batchlist, 0)
    seg_batches = np.concatenate(label_batchlist, 0)
    test_area_name = "Area_" + test_area
    train_idxs, test_idxs = [], []
    for i, room_name in enumerate(room_filelist):
        if test_area_name in room_name:
            test_idxs.append(i)
        else:
            train_idxs.append(i)
    if partition == 'train':
        all_data = data_batches[train_idxs, ...]
        all_seg = seg_batches[train_idxs, ...]
    else:
        all_data = data_batches[test_idxs, ...]
        all_seg = seg_batches[test_idxs, ...]
    return all_data, all_seg


def load_color_partseg():
    colors = []
    labels = []
    f = open("prepare_data\\meta\\partseg_colors.txt")
    for line in json.load(f):
        colors.append(line['color'])
        labels.append(line['label'])
    partseg_colors = np.array(colors)
    partseg_colors = partseg_colors[:, [2, 1, 0]]
    partseg_labels = np.array(labels)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_size = 1350
    img = np.zeros((1350, 1890, 3), dtype="uint8")
    cv2.rectangle(img, (0, 0), (1900, 1900), [255, 255, 255], thickness=-1)
    column_numbers = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    column_gaps = [320, 320, 300, 300, 285, 285]
    color_size = 64
    color_index = 0
    label_index = 0
    row_index = 16
    for row in range(0, img_size):
        column_index = 32
        for column in range(0, img_size):
            color = partseg_colors[color_index]
            label = partseg_labels[label_index]
            length = len(str(label))
            cv2.rectangle(img, (column_index, row_index), (column_index + color_size, row_index + color_size),
                          color=(int(color[0]), int(color[1]), int(color[2])), thickness=-1)
            img = cv2.putText(img, label, (column_index + int(color_size * 1.15), row_index + int(color_size / 2)),
                              font,
                              0.76, (0, 0, 0), 2)
            column_index = column_index + column_gaps[column]
            color_index = color_index + 1
            label_index = label_index + 1
            if color_index >= 50:
                cv2.imwrite("prepare_data\\meta\\partseg_colors.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                return np.array(colors)
            elif (column + 1 >= column_numbers[row]):
                break
        row_index = row_index + int(color_size * 1.3)
        if (row_index >= img_size):
            break


def load_color_semseg():
    colors = []
    labels = []
    f = open("prepare_data\\meta\\semseg_colors.txt")
    for line in json.load(f):
        colors.append(line['color'])
        labels.append(line['label'])
    semseg_colors = np.array(colors)
    semseg_colors = semseg_colors[:, [2, 1, 0]]
    partseg_labels = np.array(labels)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_size = 1500
    img = np.zeros((500, img_size, 3), dtype="uint8")
    cv2.rectangle(img, (0, 0), (img_size, 750), [255, 255, 255], thickness=-1)
    color_size = 64
    color_index = 0
    label_index = 0
    row_index = 16
    for _ in range(0, img_size):
        column_index = 32
        for _ in range(0, img_size):
            color = semseg_colors[color_index]
            label = partseg_labels[label_index]
            length = len(str(label))
            cv2.rectangle(img, (column_index, row_index), (column_index + color_size, row_index + color_size),
                          color=(int(color[0]), int(color[1]), int(color[2])), thickness=-1)
            img = cv2.putText(img, label, (column_index + int(color_size * 1.15), row_index + int(color_size / 2)),
                              font,
                              0.7, (0, 0, 0), 2)
            column_index = column_index + 200
            color_index = color_index + 1
            label_index = label_index + 1
            if color_index >= 13:
                cv2.imwrite("prepare_data\\meta\\semseg_colors.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                return np.array(colors)
            elif (column_index >= 1280):
                break
        row_index = row_index + int(color_size * 1.3)
        if (row_index >= img_size):
            break  
    

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', args=None):
        self.data, self.label = load_data_cls(partition)
        self.num_points = num_points
        self.partition = partition        
        self.PointWOLF = PointWOLF(args) if args is not None else None

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            np.random.shuffle(pointcloud)
            if self.PointWOLF is not None:
                _, pointcloud = self.PointWOLF(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


class ShapeNetPart(Dataset):
    def __init__(self, num_points, partition='train', class_choice=None):
        self.data, self.label, self.seg = load_data_partseg(partition)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition        
        self.class_choice = class_choice
        self.partseg_colors = load_color_partseg()
        
        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50 #seg_num_all表示类别数量，不选择类型，则为，选中类型，则为类型数量
            self.seg_start_index = 0 #seg_start_index表示其实索引值
            
      
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'trainval':
            # pointcloud = translate_pointcloud(pointcloud)
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        return pointcloud, label, seg

    def __len__(self):
        return self.data.shape[0]


class S3DIS(Dataset):
    def __init__(self, num_points=4096, partition='train', test_area='1'):
        self.data, self.seg = load_data_semseg(partition, test_area)
        self.num_points = num_points
        self.partition = partition    
        self.semseg_colors = load_color_semseg()

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'train':
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        seg = torch.LongTensor(seg)
        return pointcloud, seg

    def __len__(self):
        return self.data.shape[0]


class S3DISDataset(Dataset):
    def __init__(self, root, num_points, split='train', with_normalized_coords=True, holdout_area=5):
        """
        :param root: directory path to the s3dis dataset
        :param num_points: number of points to process for each scene
        :param split: 'train' or 'test'
        :param with_normalized_coords: whether include the normalized coords in features (default: True)
        :param holdout_area: which area to holdout (default: 5)
        """
        assert split in ['train', 'test']
        self.root = root
        self.split = split
        self.num_points = num_points
        self.holdout_area = None if holdout_area is None else int(holdout_area)
        self.with_normalized_coords = with_normalized_coords
        # keep at most 20/30 files in memory
        self.cache_size = 20 if split == 'train' else 30
        self.cache = {}

        # mapping batch index to corresponding file
        areas = []
        if self.split == 'train':
            for a in range(1, 7):
                if a != self.holdout_area:
                    areas.append(os.path.join(self.root, f'Area_{a}'))
        else:
            areas.append(os.path.join(self.root, f'Area_{self.holdout_area}'))

        self.num_scene_windows, self.max_num_points = 0, 0
        index_to_filename, scene_list = [], {}
        filename_to_start_index = {}
        for area in areas:
            if not os.path.isdir(os.path.join(area, scene)):
                 continue
            area_scenes = os.listdir(area)
            area_scenes.sort()
            for scene in area_scenes:
                current_scene = os.path.join(area, scene)
                scene_list[current_scene] = []
                for split in ['zero', 'half']:
                    current_file = os.path.join(current_scene, f'{split}_0.h5')
                    filename_to_start_index[current_file] = self.num_scene_windows
                    h5f = h5py.File(current_file, 'r')
                    num_windows = h5f['data'].shape[0]
                    self.num_scene_windows += num_windows
                    for i in range(num_windows):
                        index_to_filename.append(current_file)
                    scene_list[current_scene].append(current_file)
        self.index_to_filename = index_to_filename
        self.filename_to_start_index = filename_to_start_index
        self.scene_list = scene_list

    def __len__(self):
        return self.num_scene_windows

    def __getitem__(self, index):
        filename = self.index_to_filename[index]
        if filename not in self.cache.keys():
            h5f = h5py.File(filename, 'r')
            scene_data = h5f['data']
            scene_label = h5f['label_seg']
            scene_num_points = h5f['data_num']
            if len(self.cache.keys()) < self.cache_size:
                self.cache[filename] = (scene_data, scene_label, scene_num_points)
            else:
                victim_idx = np.random.randint(0, self.cache_size)
                cache_keys = list(self.cache.keys())
                cache_keys.sort()
                self.cache.pop(cache_keys[victim_idx])
                self.cache[filename] = (scene_data, scene_label, scene_num_points)
        else:
            scene_data, scene_label, scene_num_points = self.cache[filename]

        internal_pos = index - self.filename_to_start_index[filename]
        current_window_data = np.array(scene_data[internal_pos]).astype(np.float32)
        current_window_label = np.array(scene_label[internal_pos]).astype(np.int64)
        current_window_num_points = scene_num_points[internal_pos]

        # choices = np.random.choice(current_window_num_points, self.num_points,
        #                            replace=(current_window_num_points < self.num_points))
        # data = current_window_data[choices, ...].transpose()
        data = current_window_data.transpose()
        label = current_window_label
        # data[9, num_points] = [x_in_block, y_in_block, z_in_block, r, g, b, x / x_room, y / y_room, z / z_room]
        if self.with_normalized_coords:
            return data, label
        else:
            return data[:-3, :], label


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    data, label = train[0]
    print(data.shape)
    print(label.shape)

    trainval = ShapeNetPart(2048, 'trainval')
    test = ShapeNetPart(2048, 'test')
    data, label, seg = trainval[0]
    print(data.shape)
    print(label.shape)
    print(seg.shape)
    
    # train = S3DISDataset('D:\\datasets\\s3dis',4096)
    # test = S3DISDataset('D:\\datasets\\s3dis',4096, 'test')
    train=S3DIS(4096,'train')
    test=S3DIS(4096,'test')
    data, seg = train[0]
    print(data.shape)
    print(seg.shape)
