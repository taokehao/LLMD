import argparse
import os
import json
import warnings

import torch
import numpy as np
from pymatgen.core.structure import Structure

# 从 data.py 中导入需要的模块，但不调用 CIFData
from external_models.data import GaussianDistance, AtomCustomJSONInitializer
from external_models.model import CrystalGraphConvNet

# 定义归一化器，与 Pre-train.py 中一致
class Normalizer(object):
    """归一化/反归一化工具，用于还原预测值"""
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

def process_cif(cif_file, atom_init_file, radius=8, max_num_nbr=12, dmin=0, step=0.2):
    """
    读取 cif 文件并生成模型预测所需的输入张量：
      - atom_fea: 每个原子的特征，形状 (n_atoms, orig_atom_fea_len)
      - nbr_fea: 每个原子的邻居距离经过高斯展开后的特征，形状 (n_atoms, max_num_nbr, len(filter))
      - nbr_fea_idx: 每个原子的邻居索引，形状 (n_atoms, max_num_nbr)
      - crystal_atom_idx: 单个晶体时为 [0, 1, ..., n_atoms-1]
    """
    # 读取 cif 文件，确保路径正确
    if not os.path.isfile(cif_file):
        raise FileNotFoundError(f"cif 文件 {cif_file} 不存在！")
    structure = Structure.from_file(cif_file)
    
    # 检查 atom_init_file，如果传入的是目录，则自动拼接文件名
    if os.path.isdir(atom_init_file):
        atom_init_file = os.path.join(atom_init_file, "atom_init.json")
    if not os.path.isfile(atom_init_file):
        raise FileNotFoundError(f"atom_init 文件 {atom_init_file} 不存在！")
        
    # 使用 atom_init_file 构造原子初始化器
    ari = AtomCustomJSONInitializer(atom_init_file)
    
    # 计算每个原子的特征
    atom_fea_list = []
    for i in range(len(structure)):
        atomic_num = structure[i].specie.number
        fea = ari.get_atom_fea(atomic_num)
        atom_fea_list.append(fea)
    atom_fea = np.vstack(atom_fea_list)
    atom_fea = torch.tensor(atom_fea, dtype=torch.float)
    
    # 获取每个原子在给定半径内的所有邻居（包括距离和索引信息）
    all_nbrs = structure.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    
    nbr_fea_idx = []
    nbr_fea = []
    for nbr in all_nbrs:
        if len(nbr) < max_num_nbr:
            warnings.warn(f"{cif_file} 未能找到足够的邻居，已进行填充。")
            idx_list = [x[2] for x in nbr] + [0] * (max_num_nbr - len(nbr))
            dist_list = [x[1] for x in nbr] + [radius + 1.0] * (max_num_nbr - len(nbr))
        else:
            idx_list = [x[2] for x in nbr[:max_num_nbr]]
            dist_list = [x[1] for x in nbr[:max_num_nbr]]
        nbr_fea_idx.append(idx_list)
        nbr_fea.append(dist_list)
    nbr_fea_idx = np.array(nbr_fea_idx)
    nbr_fea = np.array(nbr_fea)
    
    # 使用 GaussianDistance 对邻居距离进行高斯展开
    gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)
    nbr_fea = gdf.expand(nbr_fea)
    nbr_fea = torch.tensor(nbr_fea, dtype=torch.float)
    nbr_fea_idx = torch.tensor(nbr_fea_idx, dtype=torch.long)
    
    # 对于单个晶体，crystal_atom_idx 为 [0, 1, ..., n_atoms-1]
    n_atoms = atom_fea.shape[0]
    crystal_atom_idx = [torch.arange(n_atoms, dtype=torch.long)]
    
    return atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx

def predict(cif_path):
    model_path = "./external_models/model_best-pretrain.pth.tar"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 处理 cif 文件，生成模型预测所需的输入数据
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = process_cif(
        cif_path, './external_models/atom_init.json', radius=8, max_num_nbr=12, dmin=0, step=0.2)
    
    # 获取原子特征及邻居特征的长度，用于构建模型
    orig_atom_fea_len = atom_fea.shape[-1]
    nbr_fea_len = nbr_fea.shape[-1]
    
    # 加载 checkpoint，获取保存的超参数和归一化器状态
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"模型 checkpoint 文件 {model_path} 不存在！")
    checkpoint = torch.load(model_path, map_location=device)
    saved_args = checkpoint.get('args', {})
    
    # 根据 checkpoint 中保存的参数构建模型（这里默认任务为回归，即 classification=False）
    model = CrystalGraphConvNet(
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=saved_args.get("atom_fea_len", 64),
        n_conv=saved_args.get("n_conv", 3),
        h_fea_len=saved_args.get("h_fea_len", 128),
        n_h=saved_args.get("n_h", 1),
        classification=False,
        graphormer_layers=saved_args.get("graphormer_layers", 1),
        num_heads=saved_args.get("num_heads", 4),
        max_path_distance=saved_args.get("max_path_distance", 5),
        node_dim=saved_args.get("node_dim", 128),
        edge_dim=saved_args.get("edge_dim", 128)
    )
    model.to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # 加载归一化器状态
    normalizer = Normalizer(torch.zeros(1))
    normalizer.load_state_dict(checkpoint['normalizer'])
    
    # 将数据移动到 device 上
    atom_fea = atom_fea.to(device)
    nbr_fea = nbr_fea.to(device)
    nbr_fea_idx = nbr_fea_idx.to(device)
    crystal_atom_idx = [idx.to(device) for idx in crystal_atom_idx]
    
    with torch.no_grad():
        output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
    
    # 对预测结果进行反归一化处理（回归任务输出为标量）
    pred = normalizer.denorm(output)
    print("预测结果：", pred.item())
    return pred.item()

if __name__ == "__main__":
    predict("../test/934.cif")
