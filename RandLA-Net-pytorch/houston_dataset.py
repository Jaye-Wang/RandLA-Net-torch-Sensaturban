# -*- coding: gb18030 -*-
from utils.data_process import DataProcessing as DP
from utils.config import ConfigHouston as cfg
from os.path import join
import numpy as np
import time, pickle, argparse, glob, os
from os.path import join
from helper_ply import read_ply
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch

# read the subsampled data and divide the data into training and validation
class Houston(Dataset):
    def __init__(self, mode):
        self.name = 'Houston18'
        self.mode = mode
        self.path = '/data1/gao_guang_pu'
        self.label_to_names = {0: 'Unclassified',
                               1: 'Healthy grass',
                               2: 'Stressed grass',
                               3: 'Artificial turf',
                               4: 'Evergreen trees',
                               5: 'Deciduous trees',
                               6: 'Bare earth',
                               7: 'Water',
                               8: 'Residential buildings',
                               9: 'Non-residential buildings',
                               10: 'Roads',
                               11: 'Sidewalks',
                               12: 'Crosswalks',
                               13: 'Major thoroughfares',
                               14: 'Highways',
                               15: 'Railways',
                               16: 'Paved parking lots',
                               17: 'Unpaved parking lots',
                               18: 'Cars',
                               19: 'Trains',
                               20: 'Stadium seats'
                               }
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])        # ������������,���б�ת��Ϊndarray��ʽ
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}             # {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12}
        self.ignored_labels = np.array([])                                              # ������ݼ���û��ignored��ǩ

        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]

        self.all_files = np.sort(glob.glob(join(self.path, 'train_ply', '*.ply')))
        self.train_file_name = ['block_8', 'block_9', 'block_12',
                      'block_13', 'block_16', 'block_17',
                      'block_20', 'block_21', 'block_24',
                      'block_25', 'block_28', 'block_29',
                      'block_32', 'block_33', 'block_36',
                      'block_37']
        self.test_file_name = []
        self.use_val = False 

        self.size = len(self.all_files)  

        # Initiate containers
        self.num_per_class = np.zeros(self.num_classes)
        self.val_proj = []
        self.val_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        self.load_sub_sampled_clouds(cfg.sub_grid_size)
        for ignore_label in self.ignored_labels:
            self.num_per_class = np.delete(self.num_per_class, ignore_label)
        cfg.class_weights = DP.get_class_weights(self.num_per_class, 'sqrt')

        print('Size of training : ', len(self.input_colors['training']))                # ѵ�����ж��ٸ�������112����Area2-4��
        
    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path, 'grid_{:.3f}'.format(sub_grid_size))
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            if cloud_name in self.test_file_name:
                cloud_split = 'test'
            elif cloud_name in self.train_file_name:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))            # �����ǲ����������
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            data = read_ply(sub_ply_file)                                                   # data['red'] ����ô����������һ��һά���������������red����ɫ���
            sub_colors = np.vstack((data['x'], data['y'], data['z'])).T            # �õ�һ��n*3�ľ���        
            sub_labels = data['class']
            
            if cloud_split == 'training':
                self.num_per_class += DP.get_num_class_from_label(sub_labels, self.num_classes)
                
                
            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]              # �б���б� ��ʾ �б��ƴ�ӣ�input_trees�ֵ��б����������б�ÿ���б��е�Ԫ�ض���kdtree����
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]

            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        
    def __getitem__(self, idx):
        pass

    def __len__(self):
        # Number of clouds 
        return self.size


class HoustonSampler(Dataset):

    def __init__(self, dataset, split='training'):
        self.dataset = dataset
        self.split = split
        self.possibility = {}
        self.min_possibility = {}

        if split == 'training':
            self.num_per_epoch = cfg.train_steps * cfg.batch_size       
        elif split == 'validation':
            self.num_per_epoch = cfg.val_steps * cfg.val_batch_size

        self.possibility[split] = []
        self.min_possibility[split] = []
        for i, tree in enumerate(self.dataset.input_colors[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]              # ������ɿ����� Ϊÿ��������ÿһ���㶼���ɿ�����
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]         # ѡ��ÿ����������С�����Ե��Ǹ���
        # �����������Ϊ�������ѡȡ�����е����ĵ㣬ѡȡ���ĵ��ͨ��kdtree�ҵ�������ĵ���Χ��K���㣨KNN��
        # �������ĵ㼰�ڽ����possibility������Щ���ͽ������У���ʵ�ֵ�Ĳ��ظ�ѡ��
        # possibility�ĸ��·�ʽ���������ʼֵ�Ļ������ۼ�һ��ֵ����ֵ��õ㵽���ĵ�ľ����йأ��Ҿ���Խ�󣬸�ֵԽС�����main_S3DIS��146�У���
        # ͨ����������possibility�ķ�ʽ��ʹ�ó���ĵ���к�С�Ŀ��ܱ����У��Ӷ�ʵ��������ٵ�Ŀ�ġ�

    def __getitem__(self, item):
        selected_pc, selected_labels, selected_idx, cloud_ind = self.spatially_regular_gen(item, self.split)
        return selected_pc, selected_labels, selected_idx, cloud_ind

    def __len__(self):
        
        return self.num_per_epoch
        # return 2 * cfg.val_batch_size


    def spatially_regular_gen(self, item, split):

        # Choose a random cloud         # ѡ���������С���Ǹ��������ĳ���
        cloud_idx = int(np.argmin(self.min_possibility[split]))     

        # choose the point with the minimum of possibility in the cloud as query point  ѡ��ó����µ���С���ʵĵ���Ϊ��ѯ�� point_ind�ǵ�����
        point_ind = np.argmin(self.possibility[split][cloud_idx])

        # Get all points within the cloud from tree structure   ��kdtree�еõ���������е����е��xyz����
        points = np.array(self.dataset.input_trees[split][cloud_idx].data, copy=False)

        # Center point of input region  �����е���ѡ��������͵ĵ㣨������������õģ� center_point��״Ϊ(1,3)
        center_point = points[point_ind, :].reshape(1, -1)

        # Add noise to the center point
        noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
        pick_point = center_point + noise.astype(center_point.dtype)                    # �������

        # Check if the number of points in the selected cloud is less than the predefined num_points
        if len(points) < cfg.num_points:    # ���ȡ40960����(���������г�������40960���㣬�����ľ�ȫ��ȡ����)
            # Query all points within the cloud
            queried_idx = self.dataset.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]
        else:
            # Query the predefined number of points
            queried_idx = self.dataset.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

        # Shuffle index
        queried_idx = DP.shuffle_idx(queried_idx)       # ����Ž������´��ҷ���
        # Get corresponding points and colors based on the index
        queried_pc_xyz = points[queried_idx]            # ��xyz��Ϣ���д��� ���б���Ϊ�������б����ÿ��������������У���һ���ᣩ������˳�򷵻أ����ڴ��Ҿ���
        queried_pc_xyz = queried_pc_xyz - pick_point    # ��ȥ���ĵ㣬ȥ���Ļ�
        queried_pc_colors = self.dataset.input_colors[split][cloud_idx][queried_idx]
        queried_pc_labels = self.dataset.input_labels[split][cloud_idx][queried_idx]

        # Update the possibility of the selected points
        dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)    # ����ÿ���������ĵ�ľ���
        delta = np.square(1 - dists / np.max(dists))    # ����ע���ȳ˳���Ӽ��� ������ؼ�����¸��ʵĴ�С�������ĵ�ԽԶ��Ҫ�ӵĸ��ʾ�ԽС��Խ��������һ��ѡ���ĵ�ʱ��ѡ�У�
        self.possibility[split][cloud_idx][queried_idx] += delta    # ����Ӧ���Ǹ��¸��ʣ�����һѡ���ĵ�ʱ���ظ�
        self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))  # ���¸ó�������С����

        # up_sampled with replacement
        if len(points) < cfg.num_points:    # �������40960���㣬��ʹ��������ǿ����ô�����
            queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points) 


        queried_pc_xyz = torch.from_numpy(queried_pc_xyz).float()           # ת����������ʽ
        queried_pc_colors = torch.from_numpy(queried_pc_colors).float()
        queried_pc_labels = torch.from_numpy(queried_pc_labels).long()
        queried_idx = torch.from_numpy(queried_idx).float() # keep float here?
        cloud_idx = torch.from_numpy(np.array([cloud_idx], dtype=np.int32)).float()

        points = torch.cat( (queried_pc_xyz, queried_pc_colors), 1)
    
        return points, queried_pc_labels, queried_idx, cloud_idx      


    def tf_map(self, batch_xyz, batch_features, batch_label, batch_pc_idx, batch_cloud_idx):    # �����²�����KNN��������¼��Ϊ����������׼��
        batch_features = np.concatenate([batch_xyz, batch_features], axis=-1)
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(cfg.num_layers):     # ÿһ��Ľ�����������ʵ�֣������￪ʼ��������������Ҿ����˳���ˣ���Ϊknn search�������Ǿ���������ҵ����ڵ㣩
            neighbour_idx = DP.knn_search(batch_xyz, batch_xyz, cfg.k_n)      # KNN����ÿ������Χ16���㣬��¼���������ά���ǣ�6��40960��16��
            sub_points = batch_xyz[:, :batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]      # ����²��� ά���ǣ�6��40690//4��3��
            pool_i = neighbour_idx[:, :batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]      # ������Ҳ����²��� ��6��40960//4��16��
            up_i = DP.knn_search(sub_points, batch_xyz, 1)                      # KNN����ÿ��ԭ��������²����� ά���ǣ�6��40960��1��
            input_points.append(batch_xyz)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_xyz = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [batch_features, batch_label, batch_pc_idx, batch_cloud_idx]

        return input_list

    # ���������ÿ��dataloader��һ������ִ��һ��
    def collate_fn(self,batch):

        selected_pc, selected_labels, selected_idx, cloud_ind = [],[],[],[]
        for i in range(len(batch)):
            selected_pc.append(batch[i][0])
            selected_labels.append(batch[i][1])
            selected_idx.append(batch[i][2])
            cloud_ind.append(batch[i][3])

        selected_pc = np.stack(selected_pc)                     # ���б�ѵ������γɾ���ά��Ϊ��batch��nums��feature��=��6��40960��6��
        selected_labels = np.stack(selected_labels)
        selected_idx = np.stack(selected_idx)
        cloud_ind = np.stack(cloud_ind)

        selected_xyz = selected_pc[:, :, 0:3]
        selected_features = selected_pc[:, :, 3:6]

        flat_inputs = self.tf_map(selected_xyz, selected_features, selected_labels, selected_idx, cloud_ind) # ����ֵ��һ������24���б���б�

        num_layers = cfg.num_layers
        inputs = {}
        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(torch.from_numpy(tmp).float())     # ���������б�ÿ���������ǰ������
        inputs['neigh_idx'] = []
        for tmp in flat_inputs[num_layers: 2 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())    # ���������б������ÿ���������ǰ��16���ھӵ����꣨��һ���б�û�н����²�����
        inputs['sub_idx'] = []
        for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())      # ���������б�������ÿ������������16���ھӵ�����
        inputs['interp_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())   # ���������б������ÿ�����������ÿ��ԭ���������²�����

        # inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1,2).float()   # ת����һ��
        inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).float()  # ����һ�£�Ϊ����Ӧ����linear��ά�ȣ���ת����
        inputs['labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()
        inputs['input_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 2]).long()
        inputs['cloud_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 3]).long()

        return inputs


if __name__ == '__main__':      # use to test
    dataset = Houston('training')
    dataset_train = HoustonSampler(dataset, split='training')
    dataloader = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True, drop_last=True, collate_fn=dataset_train.collate_fn)
    # dataloader = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    for data in dataloader:

        features = data['features']
        labels = data['labels']
        idx = data['input_inds']
        cloud_idx = data['cloud_inds']
        print(features.shape)
        print(labels.shape)
        print(idx.shape)
        print(cloud_idx.shape)
        break
