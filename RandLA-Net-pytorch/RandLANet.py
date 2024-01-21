# -*- coding: gb18030 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_utils as pt_utils
from utils.data_process import DataProcessing as DP
import numpy as np
from sklearn.metrics import confusion_matrix


class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        # if(config.name == 'S3DIS'):
        #     self.class_weights = DP.get_class_weights('S3DIS')
        #     self.fc0 = pt_utils.Conv1d(6, 8, kernel_size=1, bn=True)
        # else:
        #     self.class_weights = DP.get_class_weights('SemanticKITTI')
        #     self.fc0 = pt_utils.Conv1d(3, 8, kernel_size=1, bn=True)

        
        self.fc0 = nn.Linear(6, 8)
        self.fc0_acti = nn.LeakyReLU()
        self.fc0_bath = nn.BatchNorm1d(8, eps=1e-6, momentum=0.99)
        nn.init.constant_(self.fc0_bath.weight, 1.0)
        nn.init.constant_(self.fc0_bath.bias, 0)

      

        self.dilated_res_blocks = nn.ModuleList()       # LFA ����������
        d_in = 8
        for i in range(self.config.num_layers):
            d_out = self.config.d_out[i]
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out                      # ���Զ�����Ϊÿ��LFA�������2����dout(ʵ�ʵ����feature��ά����2����dout)

        d_out = d_in
        self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)       # ����1024 ���1024��MLP�����м���ǲ�mlp��

        self.decoder_blocks = nn.ModuleList()       # �ϲ��� ����������
        for j in range(self.config.num_layers):
            # if j < 4:                                       
            #     d_in = d_out + 2 * self.config.d_out[-j-2]          # -2����Ϊ���һ���ά�Ȳ���Ҫƴ�� �˶�������Ϊʵ�ʵ����ά����2����dout # din=1024+512 ά����������Ϊ������ƴ��
            #     d_out = 2 * self.config.d_out[-j-2]                 # ͨ�������������MLP�����ض�Ӧ���ά��
            # else:
            #     d_in = 4 * self.config.d_out[-5]            # ��һ��dout�������� 4*16=64����Ϊ64=32+32��������32����ƴ��
            #     d_out = 2 * self.config.d_out[-5]           # �������ά����32
            # self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True))
            
            if j < config.num_layers - 1:                                       
                d_in = d_out + 2 * self.config.d_out[-j-2]          # -2����Ϊ���һ���ά�Ȳ���Ҫƴ�� �˶�������Ϊʵ�ʵ����ά����2����dout # din=1024+512 ά����������Ϊ������ƴ��
                d_out = 2 * self.config.d_out[-j-2]                 # ͨ�������������MLP�����ض�Ӧ���ά��
            else:
                d_in = 4 * self.config.d_out[-config.num_layers]            # ��һ��dout�������� 4*16=64����Ϊ64=32+32��������32����ƴ��
                d_out = 2 * self.config.d_out[-config.num_layers]           # �������ά����32
            self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True))
            

        self.fc1 = pt_utils.Conv2d(d_out, 64, kernel_size=(1,1), bn=True)
        self.fc2 = pt_utils.Conv2d(64, 32, kernel_size=(1,1), bn=True)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = pt_utils.Conv2d(32, self.config.num_classes, kernel_size=(1,1), bn=False, activation=None)

    def forward(self, end_points):

        features = end_points['features']  # Batch*channel*npoints
        features = self.fc0(features)

        # ���������Ǻ���ĵ�
        features = self.fc0_acti(features)
        features = features.transpose(1,2)
        features = self.fc0_bath(features)

        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1 # ����һ��ά�ȣ���Ϊ��ʹ��2d��[1,1]��С�ľ��

        # ###########################Encoder############################
        f_encoder_list = []         # ���ڱ���ÿ��LFA�������������������ƴ�Ӳ���
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_blocks[i](features, end_points['xyz'][i], end_points['neigh_idx'][i])    # ��Ҫ�õ��ھӵ�����

            f_sampled_i = self.random_sample(f_encoder_i, end_points['sub_idx'][i])
            features = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)      # ��һ�ΰѻ�û������ʱ��Ҳ���ϣ�featureά��Ϊ32��32��decoder��������
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        features = self.decoder_0(f_encoder_list[-1])   # �м��ǲ�MLP

        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(features, end_points['interp_idx'][-j - 1])                 # �Ƚ����˲�ֵ
            f_decoder_i = self.decoder_blocks[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))        # ��֮ǰ����������ƴ��

            features = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################

        features = self.fc1(features)
        features = self.fc2(features)
        features = self.dropout(features)
        features = self.fc3(features)
        f_out = features.squeeze(3)

        end_points['logits'] = f_out
        return end_points

    @staticmethod
    def random_sample(feature, pool_idx):       # �����Ѿ�����������ֵ�������������ֻ�Ƕ�ȡ����ֵ
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)    # batch*channel*npoints   # ����һ��ά��
        num_neigh = pool_idx.shape[-1]      # knn�ھӵ�����
        d = feature.shape[1]                # ����ά��
        batch_size = pool_idx.shape[0]      # pool_idx��ά����[6, 10240, 16] ���16��16���ھӵ�����
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))  # �õ�������������
        # �����ȶ�pool_idx����һ���м��featureά�ȣ������Ϊ[batch, 1, npoints*nsamples]
        # ������������ά���Ͻ�ÿ��batch�е�ÿһ�е����ݷֱ�repeat feature.shape[1]-1 �Σ��ﵽfeature.shape[1]ά�� repeat��ά���� [batch, feature.shape[1], npoints*nsamples]
        # Ȼ������������֮���pool_idx����feature����������
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1  [0]��ȡֵ����˼ [1]������ max����˼����ÿһά������ȡ16���ڵ���������������
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1,feature.shape[1],1))  # �ҵ�Ҫ�ϲ������ĵ������
        #���Ҿ��ùؼ����������ݾ���������ԣ��ſ��Խ�����������ԭ����һ�β���ǰ�ĵ㣩
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features



def compute_acc(end_points):

    logits = end_points['valid_logits']
    labels = end_points['valid_labels']
    logits = logits.max(dim=1)[1]
    acc = (logits == labels).sum().float() / float(labels.shape[0])
    end_points['acc'] = acc
    return acc, end_points


class IoUCalculator:
    def __init__(self, cfg):
        self.gt_classes = [0 for _ in range(cfg.num_classes)]               # ��ʼ��һ������Ϊnum_classes��Ԫ��ȫΪ0���б�
        self.positive_classes = [0 for _ in range(cfg.num_classes)]         # ͬ��  
        self.true_positive_classes = [0 for _ in range(cfg.num_classes)]
        self.cfg = cfg

    def add_data(self, end_points):
        logits = end_points['valid_logits']     # ������label֮���logit        # ά���ǣ�40960*batch_size��
        labels = end_points['valid_labels']     # ������label֮���label
        pred = logits.max(dim=1)[1]             # [1] ��ѡ�����max����ĵڶ���λ�ã����max���󳤶�Ϊ������һ��λ�ô��ȡmax֮���ֵ���ڶ���λ�ô��maxֵ������
        pred_valid = pred.detach().cpu().numpy()
        labels_valid = labels.detach().cpu().numpy()

        val_total_correct = 0       # �����������ûʲô�ã�
        val_total_seen = 0

        correct = np.sum(pred_valid == labels_valid)    # ���������ȷ�ĵ���
        val_total_correct += correct    # �ۼ���ȷ�ĵ�
        val_total_seen += len(labels_valid) # �ۼ�һ���ĵ�

        # ����������󣨻������������Ԥ�����������ʵ�������������ȷ����������ĸ�����
        conf_matrix = confusion_matrix(labels_valid, pred_valid, labels=np.arange(0, self.cfg.num_classes, 1)) 
        self.gt_classes += np.sum(conf_matrix, axis=1)      # ���м���������ʾĳ�����һ���ж��ٸ���ʵ�����ݵ㣨ground truth��
        self.positive_classes += np.sum(conf_matrix, axis=0)    # ���м���������ʾĳ�����Ԥ������ٸ����ݵ�
        self.true_positive_classes += np.diagonal(conf_matrix)  # ȡ���Խ����ϵ�Ԫ��

    def compute_iou(self):
        iou_list = []
        for n in range(0, self.cfg.num_classes, 1):
            if float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n]) != 0:       # ������Ƿ�ĸ����֤��ĸ��Ϊ��
                iou = self.true_positive_classes[n] / float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n])  # ���n�����IoU
                iou_list.append(iou)
            else:
                iou_list.append(0.0)            # ����ͬʱΪ����п��ܷ�ĸΪ�㣬����iou=0
        mean_iou = sum(iou_list) / float(self.cfg.num_classes)  # ���������
        return mean_iou, iou_list



class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.mlp1 = pt_utils.Conv2d(d_in, d_out//2, kernel_size=(1,1), bn=True)
        self.lfa = Building_block(d_out)
        self.mlp2 = pt_utils.Conv2d(d_out, d_out*2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = pt_utils.Conv2d(d_in, d_out*2, kernel_size=(1,1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)  # Batch*channel*npoints*1                # ͼ����ɫ���Ǹ�MLP
        f_pc = self.lfa(xyz, f_pc, neigh_idx)  # Batch*d_out*npoints*1      # ���lfa���������ֲ��ռ���������ע�����ػ�
        f_pc = self.mlp2(f_pc)                                              # ������Ǹ���ɫ��MLP
        shortcut = self.shortcut(feature)                                   # ������Ǹ�MLP
        return F.leaky_relu(f_pc+shortcut, negative_slope=0.2)              # ��Ԫ�ؼ�����


class Building_block(nn.Module):
    def __init__(self, d_out):  #  d_in = d_out//2
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(10, d_out//2, kernel_size=(1,1), bn=True)
        self.att_pooling_1 = Att_pooling(d_out, d_out//2)

        self.mlp2 = pt_utils.Conv2d(d_out//2, d_out//2, kernel_size=(1, 1), bn=True)
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):  # feature: Batch*channel*npoints*1
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)  # batch*npoint*nsamples*10  # ����10��feature�ǹ̶���
        f_xyz = f_xyz.permute((0, 3, 1, 2))  # batch*10*npoint*nsamples  # ����������ά��
        f_xyz = self.mlp1(f_xyz)            # ���ռ��������б���,��Ӧͼ��position encoding
        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel �õ�K���ٽ����feature
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples ����ά��
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)      # ��������Ϣ�Ϳռ���Ϣƴ����һ��
        f_pc_agg = self.att_pooling_1(f_concat)  # Batch*channel*npoints*1

        f_xyz = self.mlp2(f_xyz)        # ֱ�����ϴα���õĿռ���Ϣ�ٽ���һ�α���
        f_neighbours = self.gather_neighbour(f_pc_agg.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples ����ά��
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)  # batch*npoint*nsamples*3

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)  # batch*npoint*nsamples*3  ��һ�����ƹ㲥�Ĳ�����ʹ����һ�п���ֱ����� ��һ���Ľ�������ĵ��Լ���xyz�����Ӧ�����е�pi
        relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3   # �Լ������ȥ������������������
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))  # batch*npoint*nsamples*1   # �����ĵ����Ծ���
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)  # batch*npoint*nsamples*10
        return relative_feature

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel(xyz����feature)
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)      # ���gather��������Ƚ��ѣ�Ҫ������
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))     # ��ԭʼ���xyz���꣨��feature���У��ҵ�16�����ڵ�����꣨��feature����ע�����pc����������ģ�������ֵ��neighbor_idx�й�ϵ��
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel     # �������40960�����и������16���ڵ�����
        return features


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)     # ע�����ػ����滹��һ��mlp���Ըı��������״

    def forward(self, feature_set):

        att_activation = self.fc(feature_set)           # ��ƴ�Ӻ�ľ��󾭹�һ��ȫ���Ӽ�softmaxѧϰһ����ͬά�ȵ�ע��������
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores                # ������Ԫ�����
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)   # ���
        f_agg = self.mlp(f_agg)                         # ���һ��mlp����ά��
        return f_agg


def compute_loss(end_points, cfg, device):

    logits = end_points['logits']       # �������л�ȡlogit��label
    labels = end_points['labels']

    logits = logits.transpose(1, 2).reshape(-1, cfg.num_classes)        # ��logit��label��batchά����ȥ�����·ŵ�����Ŀ��ά��
    labels = labels.reshape(-1)


    # Boolean mask of points that should be ignored
    # ignored_bool = labels == 0                              # Ӧ��������0��mask��
    # for ign_label in cfg.ignored_label_inds:
    #     ignored_bool = ignored_bool | (labels == ign_label)

    # ignored_bool = labels == 0                              
    ignored_bool = torch.zeros(len(labels), dtype=torch.bool).to(device)                         # ����û�����⣬��������Ǻ���
    ignored_bool = ignored_bool | (labels == 0)

    # Collect logits and labels that are not ignored
    valid_idx = ignored_bool == 0
    valid_logits = logits[valid_idx, :]
    valid_labels_init = labels[valid_idx]

    # Reduce label values in the range of logit shape
    reducing_list = torch.arange(0, cfg.num_classes).long().to(device)       
    inserted_value = torch.zeros((1,)).long().to(device)
    for ign_label in cfg.ignored_label_inds:
        reducing_list = torch.cat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
    valid_labels = torch.gather(reducing_list, 0, valid_labels_init)            # �������û����
    loss = get_loss(valid_logits, valid_labels, cfg.class_weights, device)
    end_points['valid_logits'], end_points['valid_labels'] = valid_logits, valid_labels     # valid_logits��ignore label֮���logit
    end_points['loss'] = loss
    return loss, end_points


def get_loss(logits, labels, pre_cal_weights, device):
    # calculate the weighted cross entropy according to the inverse frequency
    class_weights = torch.from_numpy(pre_cal_weights).float().to(device)
    # one_hot_labels = F.one_hot(labels, self.config.num_classes)

    criterion = nn.CrossEntropyLoss(weight=class_weights.reshape([-1]), reduction='none')   # �������һ��ά�ȣ��°汾pytorch��Ҫһά��Ȩ������
    output_loss = criterion(logits, labels)
    output_loss = output_loss.mean()
    return output_loss