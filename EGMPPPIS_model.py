import pickle
import dgl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore")

# 全局配置与路径
Feature_Path = "./Feature/"
SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# 版本控制
BASE_MODEL_TYPE = 'EGMP'  # w/o_CoordUpdate  w/o_GlobalCtx  w/o_ResFusion
ADD_NODEFEATS = 'all'

MAP_CUTOFF = 14
DIST_NORM = 15
NUM_CLASSES = 2
BATCH_SIZE = 1
NUMBER_EPOCHS = 50

# 去掉ESM2后，序列/进化特征只保留 PSSM 20 + HMM 20 = 40
ESM2_DIM = 0
PSSM_DIM = 20
HMM_DIM = 20
SEQ_DIM = PSSM_DIM + HMM_DIM  # 40

# DSSP 14 + Atom/resAF 7 + PsePos 1 = 22
STRUCT_DIM = 22
INPUT_DIM = SEQ_DIM + STRUCT_DIM  # 40 + 22 = 62

HIDDEN_DIM = 256
LAYER = 4
DROPOUT = 0.5
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4


# 数据处理与加载
def get_dssp_features(sequence_name):
    return np.load(Feature_Path + "dssp/" + sequence_name + '.npy')


def embedding(sequence_name):
    """
    返回 61 维基础残基特征：
    PSSM(20) + HMM(20) + DSSP(14) + Atom/resAF(7)
    第 62 维 PsePos 在 Dataset.__getitem__ 中拼接。
    """
    # 1. 传统进化特征
    pssm_feat = np.load(Feature_Path + "pssm/" + sequence_name + '.npy')
    hmm_feat = np.load(Feature_Path + "hmm/" + sequence_name + '.npy')

    # 2. 结构属性特征
    dssp_feat = get_dssp_features(sequence_name)
    atom_feat = np.load(Feature_Path + "resAF/" + sequence_name + '.npy')

    return np.concatenate([pssm_feat, hmm_feat, dssp_feat, atom_feat], axis=1).astype(np.float32)


def load_graph(sequence_name):
    dismap = np.load(Feature_Path + "distance_map_SC/" + sequence_name + ".npy")
    mask = ((dismap >= 0) * (dismap <= MAP_CUTOFF))
    adj = mask.astype(int)
    radius_index = np.where(adj == 1)
    g = dgl.graph((radius_index[0], radius_index[1]))
    return adj, g, radius_index


class ProDataset(Dataset):
    def __init__(self, dataframe, radius=MAP_CUTOFF, dist=DIST_NORM,
                 psepos_path='./Feature/psepos/Train335_psepos_SC.pkl'):
        self.df = dataframe
        self.psepos_dict = pickle.load(open(psepos_path, 'rb'))
        self.radius = radius
        self.dist = dist

    def __len__(self):
        return len(self.df)

    def cal_edge_attr(self, index_list, pos):
        pdist = nn.PairwiseDistance(p=2, keepdim=True)
        cossim = nn.CosineSimilarity(dim=1)
        distance = (pdist(pos[index_list[0]], pos[index_list[1]]) / self.radius).detach().numpy()
        cos = ((cossim(pos[index_list[0]], pos[index_list[1]]).unsqueeze(-1) + 1) / 2).detach().numpy()
        return np.array([distance, cos])

    def __getitem__(self, idx):
        protein_id = self.df.iloc[idx]['ID']
        seq = self.df.iloc[idx]['sequence']
        label = torch.tensor(self.df.iloc[idx]['label'], dtype=torch.long)

        adj, g, radius_index_list = load_graph(protein_id)

        # 伪坐标：以第一个残基为参考点，使用相对坐标
        pos_raw = self.psepos_dict[protein_id].astype(np.float32)
        reference_pos = pos_raw[0]
        pos_relative = pos_raw - reference_pos
        pos_tensor = torch.from_numpy(pos_relative)
        g.ndata['pos'] = pos_tensor

        if len(radius_index_list[0]) > 0:
            edge_feat = self.cal_edge_attr(radius_index_list, pos_tensor)
            edge_feat = np.transpose(edge_feat, (1, 2, 0)).squeeze(1)
        else:
            edge_feat = np.zeros((0, 2), dtype=np.float32)
        g.edata['ex'] = torch.tensor(edge_feat, dtype=torch.float32)

        # 基础特征61维 + PsePos 1 维 = 62维
        base_feat = torch.tensor(embedding(protein_id), dtype=torch.float32)
        psepos_1d = torch.sqrt(torch.sum(pos_tensor * pos_tensor, dim=1)).unsqueeze(-1) / self.dist
        feat = torch.cat([base_feat, psepos_1d], dim=-1)

        return protein_id, seq, label, feat, g, torch.tensor(adj, dtype=torch.float32)


def graph_collate(batch):
    ids, seqs, labels, feats, graphs, adjs = zip(*batch)
    return ids, seqs, torch.cat(labels), torch.cat(feats), dgl.batch(graphs), torch.stack(adjs)


# 核心层
class EGMP_Layer(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.h_dim = h_dim

        self.edge_net = nn.Sequential(
            nn.Linear(h_dim * 2 + 1, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU()
        )

        # 1. 坐标更新消融控制
        if BASE_MODEL_TYPE != 'w/o_CoordUpdate':
            self.coord_net = nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.SiLU(),
                nn.Linear(h_dim, 1, bias=False)
            )
            torch.nn.init.xavier_uniform_(self.coord_net[-1].weight, gain=0.001)

        # 2. 全局上下文消融控制：动态调整输入维度
        node_in_dim = h_dim * 2 if BASE_MODEL_TYPE == 'w/o_GlobalCtx' else h_dim * 3

        self.node_net = nn.Sequential(
            nn.Linear(node_in_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim)
        )
        self.attn_net = nn.Sequential(
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

    def message_func(self, edges):
        coord_diff = edges.src['pos'] - edges.dst['pos']
        dist_sq = torch.sum(coord_diff ** 2, dim=-1, keepdim=True)
        h_cat = torch.cat([edges.src['h'], edges.dst['h'], dist_sq], dim=-1)

        m_ij = self.edge_net(h_cat)
        attn = self.attn_net(m_ij)
        m_ij = m_ij * attn

        if BASE_MODEL_TYPE == 'w/o_CoordUpdate':
            return {'m': m_ij}

        pos_update = coord_diff * self.coord_net(m_ij)
        return {'m': m_ij, 'pos_update': pos_update}

    def reduce_func(self, nodes):
        m_agg = torch.sum(nodes.mailbox['m'], dim=1)

        if BASE_MODEL_TYPE == 'w/o_CoordUpdate':
            return {'m_agg': m_agg}

        pos_new = torch.mean(nodes.mailbox['pos_update'], dim=1)
        return {'m_agg': m_agg, 'pos_new': pos_new}

    def forward(self, g, h, pos):
        with g.local_scope():
            g.ndata['h'] = h
            g.ndata['pos'] = pos
            g.update_all(self.message_func, self.reduce_func)

            node_inputs = [h, g.ndata['m_agg']]

            # 全局上下文消融控制
            if BASE_MODEL_TYPE != 'w/o_GlobalCtx':
                global_context = dgl.mean_nodes(g, 'h')
                global_context_broadcast = dgl.broadcast_nodes(g, global_context)
                node_inputs.append(global_context_broadcast)

            node_input = torch.cat(node_inputs, dim=-1)
            h_out = h + self.node_net(node_input)

            if BASE_MODEL_TYPE == 'w/o_CoordUpdate':
                pos_out = pos
            else:
                pos_out = pos + g.ndata['pos_new']

            return h_out, pos_out


class EGMPPPIS(nn.Module):

    def __init__(self, nlayers=LAYER, nfeat=None, nhidden=HIDDEN_DIM, nclass=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()

        self.seq_proj = nn.Linear(SEQ_DIM, HIDDEN_DIM)
        self.struct_proj = nn.Linear(STRUCT_DIM, HIDDEN_DIM)

        self.layers = nn.ModuleList([EGMP_Layer(HIDDEN_DIM) for _ in range(nlayers)])
        self.dropout = nn.Dropout(dropout)

        # 3. 残差融合消融控制：动态调整分类器输入维度
        classifier_in_dim = HIDDEN_DIM if BASE_MODEL_TYPE == 'w/o_ResFusion' else HIDDEN_DIM * 2

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, HIDDEN_DIM),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_DIM, nclass)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.6,
            patience=10,
            min_lr=1e-6
        )

    def forward(self, x, graph, adj_matrix):
        x = x.view(-1, x.shape[-1])

        # 前40维：PSSM + HMM
        h_seq = F.silu(self.seq_proj(x[:, :SEQ_DIM]))

        # 后22维：DSSP + Atom + PsePos
        h_struct = F.silu(self.struct_proj(x[:, SEQ_DIM:]))

        # 原始加性融合
        h_fused = h_seq + h_struct
        pos = graph.ndata['pos']

        for layer in self.layers:
            h_fused, pos = layer(graph, h_fused, pos)
            h_fused = self.dropout(h_fused)

        # 残差融合消融控制
        if BASE_MODEL_TYPE == 'w/o_ResFusion':
            h_final = h_fused
        else:
            h_final = torch.cat([h_seq, h_fused], dim=-1)

        return self.classifier(h_final)