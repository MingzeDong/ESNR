from . import sinkhorn_knopp as skp
from torch_geometric.utils import to_dense_adj
import numpy as np
from torch.nn.functional import one_hot
import torch
from sklearn.cluster import AgglomerativeClustering
from torch_geometric.utils import add_self_loops

def esnr_MP(data,eps = 1e-3, thres=1,seed=42, return_num = False):
    torch.manual_seed(seed)
    X = np.array(data.x)
    cluster = AgglomerativeClustering(n_clusters=round(np.sqrt(X.shape[0])))
    cluster.fit(X)

    S = to_dense_adj(data.edge_index,max_num_nodes=data.x.shape[0]).squeeze()
    L = one_hot(torch.from_numpy(cluster.labels_))
    #L = one_hot(torch.arange(0, S.shape[0]) % round(np.sqrt(S.shape[0])))
    #idx = torch.randperm(S.shape[0])
    #L = L[idx,:]
    LS = L.T.float() @ S
    LS = LS.numpy() + eps
    sk = skp.SinkhornKnopp()
    P = sk.fit(LS)
    wm = np.dot(np.dot(np.sqrt(sk._D1),LS),np.sqrt(sk._D2))
    u,s,vt = np.linalg.svd(wm-np.mean(wm))
    if np.sum(s > ((1+thres) * (np.sqrt(LS.shape[0])+np.sqrt(LS.shape[1])))) > 0:
        bd = ((1+thres) * (np.sqrt(LS.shape[0])+np.sqrt(LS.shape[1])))
        s_ = np.mean(((s-bd) * (s > bd))/s)
        r_ = np.sum(s > ((1+thres) * (np.sqrt(LS.shape[0])+np.sqrt(LS.shape[1])))) / (L.shape[1]-1)
    else:
        s_ = 0
        r_ = 0

    if return_num:
        return LS, s*s/LS.shape[1], LS.shape[0]/LS.shape[1], r_
    else:
        return r_


def esnr_nc(data,eps = 1e-3, thres=1,seed=42,max_iter=1000, return_num = False):
    torch.manual_seed(seed)
    X = np.array(data.x)

    S = to_dense_adj(data.edge_index,max_num_nodes=data.x.shape[0]).squeeze()
    L = one_hot(data.y.squeeze())
    #L = one_hot(torch.arange(0, S.shape[0]) % round(np.sqrt(S.shape[0])))
    #idx = torch.randperm(S.shape[0])
    #L = L[idx,:]
    LS = L.T.float() @ S
    LS = LS.numpy() + eps
    sk = skp.SinkhornKnopp(max_iter=max_iter)
    P = sk.fit(LS)
    wm = np.dot(np.dot(np.sqrt(sk._D1),LS),np.sqrt(sk._D2))
    u,s,vt = np.linalg.svd(wm-np.mean(wm))
    if np.sum(s > ((1+thres) * (np.sqrt(LS.shape[0])+np.sqrt(LS.shape[1])))) > 0:
        bd = ((1+thres) * (np.sqrt(LS.shape[0])+np.sqrt(LS.shape[1])))
        s_ = np.mean(((s-bd) * (s > bd))/s)
        r_ = np.sum(s > ((1+thres) * (np.sqrt(LS.shape[0])+np.sqrt(LS.shape[1])))) / (L.shape[1]-1)
    else:
        s_ = 0
        r_ = 0
    if return_num:
        return LS, s*s/LS.shape[1], LS.shape[0]/LS.shape[1], r_
    else:
        return r_, s_


def esnr_nc_directed(data,eps = 1e-3, thres=1,seed=42, return_num = False):
    torch.manual_seed(seed)
    X = np.array(data.x)

    S = to_dense_adj(data.edge_index,max_num_nodes=data.x.shape[0]).squeeze()
    S = S * 0
    S[data.edge_index[0,:],data.edge_index[1,:]] = 1
    L = one_hot(data.y)
    #L = one_hot(torch.arange(0, S.shape[0]) % round(np.sqrt(S.shape[0])))
    #idx = torch.randperm(S.shape[0])
    #L = L[idx,:]
    LS = L.T.float() @ S @ L.float()
    LS = LS.numpy() + eps
    sk = skp.SinkhornKnopp()
    P = sk.fit(LS)
    wm = np.dot(np.dot(np.sqrt(sk._D1),LS),np.sqrt(sk._D2))
    u,s,vt = np.linalg.svd(wm-np.mean(wm))
    if np.sum(s > ((1+thres) * (np.sqrt(LS.shape[0])+np.sqrt(LS.shape[1])))) > 0:
        bd = ((1+thres) * (np.sqrt(LS.shape[0])+np.sqrt(LS.shape[1])))
        s_ = np.mean(((s-bd) * (s > bd))/s)
        r_ = np.sum(s > ((1+thres) * (np.sqrt(LS.shape[0])+np.sqrt(LS.shape[1])))) / (L.shape[1]-1)
    else:
        s_ = 0
        r_ = 0
    if return_num:
        return LS, s*s/LS.shape[1], LS.shape[0]/LS.shape[1], r_
    else:
        return r_, s_

def esnr_vanilla(data):

    S = to_dense_adj(data.edge_index,max_num_nodes=data.x.shape[0]).squeeze()
    L = one_hot(data.y)
    LS = L.T.float() @ S
    s_ =  (torch.sum(LS * LS) - torch.sum(LS))/torch.sum(LS)

    return s_.cpu().numpy()

def aggre_homophily(data):
    A = to_dense_adj(add_self_loops(data.edge_index)[0],max_num_nodes=data.x.shape[0]).squeeze()
    Ahat = (1/A.sum(axis=1)) * A
    inner_prod = (Ahat @ data.x) @ (Ahat @ data.x).T
    weight_matrix = torch.zeros(
        A.clone().detach().size(0), data.y.clone().detach().max() + 1
    )
    for i in range(data.y.max() + 1):
        weight_matrix[:, i] = torch.mean(inner_prod[:, data.y.squeeze() == i], 1)
    x = torch.mean(torch.argmax(weight_matrix, 1).eq(data.y.squeeze()).float()).cpu().detach().numpy()
    #return (2*x-1) * ((2*x-1)>0)
    return x