import copy
import torch
import pandas as pd
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import Parameter
from typing import Union, Tuple, Optional
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_scatter import scatter_max
import scanpy as sc
# cudnn.deterministic = True
# cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class TimePredictionNetwork(nn.Module):
    def __init__(self, input_dim):
        super(TimePredictionNetwork, self).__init__()
        hidden_dim = input_dim // 2  # Use half of input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        time_pred = self.fc2(x)
        return torch.sigmoid(time_pred)

class ClusterPredictionNetwork(nn.Module):
    def __init__(self, input_dim, num_cluster):
        super(ClusterPredictionNetwork, self).__init__()
        hidden_dim = input_dim // 2  # Use half of input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_cluster)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        return self.fc2(x)


class RegulationModel(nn.Module):
    def __init__(self, type_features):
        super(RegulationModel, self).__init__()
        self.register_buffer('type_features', torch.tensor(type_features, dtype=torch.float32))
        self.learnable_matrix = nn.Parameter(torch.tensor(type_features, dtype=torch.float32))

    def forward(self, use_initial=True):
        if use_initial:
            return self.type_features # / self.type_features.shape[1]
        return self.learnable_matrix

# class Good_RegulationModel(nn.Module):
#     def __init__(self, type_features, k):
#         super(RegulationModel, self).__init__()
#         self.register_buffer('type_features', torch.tensor(type_features, dtype=torch.float32))
#         self.learnable_matrix = nn.Parameter(torch.ones(k, type_features.shape[1]))

#     def forward(self, c, use_initial=True):
#         if use_initial:
#             return self.type_features / self.type_features.shape[1]
#         # # Version 2
#         # # Compute the normalized and signed regulation matrix
#         # abs_learnable_matrix = torch.abs(self.learnable_matrix)
#         # norm_matrix = F.softmax(abs_learnable_matrix, dim=1)
#         # signed_norm_matrix = self.learnable_matrix.sign() * norm_matrix

#         # # Directly compute the adjusted regulation in a single matrix multiplication
#         # adjusted_regulation = torch.mm(c, signed_norm_matrix) * self.type_features
#         # Version 1
#         # adjusted_regulation = torch.mm(c, torch.tanh(self.learnable_matrix)) * self.type_features
#         # Version 3
#         # Compute the normalized and signed regulation matrix
#         abs_learnable_matrix = torch.abs(self.learnable_matrix)
#         norm_matrix = F.softmax(torch.mm(c, abs_learnable_matrix), dim=1)
#         signed_norm_matrix = torch.mm(c, self.learnable_matrix).sign() * norm_matrix

#         # Directly compute the adjusted regulation in a single matrix multiplication
#         adjusted_regulation = signed_norm_matrix * self.type_features
#         # # Version 4
#         # # Compute the normalized and signed regulation matrix
#         # norm_matrix = F.normalize(torch.mm(c, self.learnable_matrix), p=2, dim=1)
#         # # Directly compute the adjusted regulation in a single matrix multiplication
#         # adjusted_regulation = norm_matrix * self.type_features
#         return adjusted_regulation

class ExpertModel(nn.Module):
    def __init__(self, ein_features, eout_features, mode='original', ehidden_size=256):
        super(ExpertModel, self).__init__()
        self.mode = mode
        
        if mode == 'slim':
            # === 新版本：瘦身 + Dropout (更稳定) ===
            self.network = nn.Sequential(
                nn.Linear(ein_features + 1, ehidden_size), # 129 -> 256
                nn.LayerNorm(ehidden_size),
                nn.LeakyReLU(),
                nn.Dropout(0.2),  # 防止过拟合
                
                nn.Linear(ehidden_size, ehidden_size),     # 256 -> 256
                nn.LayerNorm(ehidden_size),
                nn.LeakyReLU(),
                nn.Dropout(0.2),

                nn.Linear(ehidden_size, eout_features),    # 256 -> 6000
                nn.Sigmoid()
            )
        else:
            # === 旧版本：原始大网络 (保持结果一致) ===
            # Default: ehidden_size calc logic form original code
            ehidden_size = 2 * ein_features
            second_layer_size = 4 * ein_features
            third_layer_size = 8 * ein_features
            
            self.network = nn.Sequential(
                nn.Linear(ein_features + 1, ehidden_size),
                nn.LayerNorm(ehidden_size),
                nn.LeakyReLU(),
                
                nn.Linear(ehidden_size, second_layer_size),
                nn.LayerNorm(second_layer_size),
                nn.LeakyReLU(),
                
                nn.Linear(second_layer_size, third_layer_size),
                nn.LayerNorm(third_layer_size),
                nn.LeakyReLU(),
                
                nn.Linear(third_layer_size, eout_features),
                nn.Sigmoid()
            )
    
    def forward(self, x, t):
        x = torch.cat((x, t), dim=1)
        return self.network(x)

class GATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.lin_src = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_normal_(self.lin_src.data)#, gain=1.414)
        #nn.init.kaiming_normal_(self.lin_src.data, a=self.negative_slope)
        self.lin_dst = self.lin_src


        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))
        nn.init.xavier_normal_(self.att_src.data)#, gain=1.414)
        #nn.init.kaiming_normal_(self.att_src.data, a=self.negative_slope)
        nn.init.xavier_normal_(self.att_dst.data)#, gain=1.414)
        #nn.init.kaiming_normal_(self.att_dst.data, a=self.negative_slope)

        self._alpha = None
        self.attentions = None

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None, attention=True, tied_attention = None):

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            # x_src = x_dst = self.lin_src(x).view(-1, H, C)
            x_src = x_dst = torch.mm(x, self.lin_src).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)
        
        x = (x_src, x_dst)

        if not attention:
            return x[0].mean(dim=1)
            # return x[0].view(-1, self.heads * self.out_channels)

        if tied_attention == None:
            alpha_src = (x_src * self.att_src).sum(dim=-1)
            alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
            alpha = (alpha_src, alpha_dst)
            self.attentions = alpha
        else:
            alpha = tied_attention

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = torch.sigmoid(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

class GATModel(nn.Module):
    def __init__(self, in_features, hid_features, out_features, num_clusters, num_genes, type_features = None, use_experts=True, expert_mode='original'):
        super(GATModel, self).__init__()
        
        # Encoder
        self.gat_conv1 = GATConv(in_features, hid_features, heads=1, concat=False,
                             dropout=0, add_self_loops=False)
        self.batch_norm1 = nn.BatchNorm1d(hid_features)  # Batch normalization for hidden features
        self.gat_conv2 = GATConv(hid_features, out_features,  heads=1, concat=False,
                             dropout=0, add_self_loops=False)
        self.batch_norm2 = nn.BatchNorm1d(out_features)  # Batch normalization for output features
        
        # Decoder
        self.gat_conv3 = GATConv(out_features, hid_features,  heads=1, concat=False,
                             dropout=0, add_self_loops=False)
        self.batch_norm3 = nn.BatchNorm1d(hid_features)  # Batch normalization for hidden features
        self.gat_conv4 = GATConv(hid_features, in_features,  heads=1, concat=False,
                             dropout=0, add_self_loops=False)
        self.batch_norm4 = nn.BatchNorm1d(in_features)  # Batch normalization for input features
        
        # # Learnable cluster-latent matrix
        # self.L_1 = nn.Parameter(torch.empty(out_features, num_clusters))
        # nn.init.xavier_normal_(self.L_1)

        # Initialize expert networks only if use_experts is True
        self.num_genes = num_genes
        self.use_experts = use_experts
        self.expert_mode = expert_mode  # 保存 mode

        if use_experts:
            self.init_experts(out_features, num_clusters, num_genes)
               
        # Time prediction network
        self.time_prediction_network = TimePredictionNetwork(out_features)
        self.cluster_network = ClusterPredictionNetwork(out_features, num_clusters)
        # self.up_regulation = RegulationModel(type_features, num_clusters)
        self.up_regulation = RegulationModel(type_features)

    def init_experts(self, out_features, num_clusters, num_genes):
        self.experts = nn.ModuleList([ExpertModel(out_features, 3 * num_genes) for _ in range(num_clusters)])

    def encode(self, x, edge_index):
        x = self.gat_conv1(x, edge_index)
        x = self.batch_norm1(x)  # Apply batch normalization after first GAT layer
        x = F.elu(x)
        x = self.gat_conv2(x, edge_index, attention=False)
        x = self.batch_norm2(x)  # Apply batch normalization after second GAT layer
        return x

    def decode(self, x, edge_index):
        self.gat_conv3.lin_src.data = self.gat_conv2.lin_src.transpose(0, 1)
        self.gat_conv3.lin_dst.data = self.gat_conv2.lin_dst.transpose(0, 1)
        self.gat_conv4.lin_src.data = self.gat_conv1.lin_src.transpose(0, 1)
        self.gat_conv4.lin_dst.data = self.gat_conv1.lin_dst.transpose(0, 1)
        x = self.gat_conv3(x, edge_index, attention=True,
                           tied_attention=self.gat_conv1.attentions)
        x = self.batch_norm3(x)  # Apply batch normalization after third GAT layer
        x = F.elu(x)
        x = self.gat_conv4(x, edge_index, attention=False)
        x = self.batch_norm4(x)  # Apply batch normalization after fourth GAT layer
        x = torch.sigmoid(x)
        return x

    # def compute_cluster_matrix(self, z):
    #     c = torch.matmul(z, self.L_1)
    #     return c

    # V2的版本
    # def compute_cell_params(self, c, z, t, num_genes):
    #     if self.use_experts:
    #         expert_outputs = torch.stack([expert(z, t) for expert in self.experts], dim=1)
    #         weights = c.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1))
    #         weighted_outputs = expert_outputs * weights
    #         summed_outputs = torch.sum(weighted_outputs, dim=1)
    #         return summed_outputs
    #     else:
    #         # Return a placeholder or default value if experts are not used
    #         return torch.zeros(z.size(0), 3 * num_genes).to(z.device)
        
    def compute_cell_params(self, c, z, t, num_genes):
        if self.use_experts:
            batch_size = z.size(0)
            output_dim = 3 * num_genes
            summed_outputs = torch.zeros(batch_size, output_dim, device=z.device)
            for i, expert in enumerate(self.experts):
                expert_out = expert(z, t)
                weight = c[:, i].unsqueeze(1) 
                summed_outputs += weight * expert_out
            return summed_outputs
        else:
            return torch.zeros(z.size(0), 3 * num_genes).to(z.device)

    def forward(self, x, edge_index, use_initial_regulation=True):
        z = self.encode(x, edge_index)
        recon = self.decode(z, edge_index)
        # c_softmax = F.softmax(self.compute_cluster_matrix(z), dim=-1)  # Softmax to get cluster assignments  
        c_softmax = F.softmax(self.cluster_network(z), dim=-1)
        # Predict time point
        t = self.time_prediction_network(z)
        # Compute cell parameters including time point
        cp = self.compute_cell_params(c_softmax, z, t, self.num_genes)
        up_type_features = self.up_regulation(use_initial=use_initial_regulation)
        
        return recon, z, t, c_softmax, cp, up_type_features

def mincut_loss(adj, c):
    k = c.size(-1)
    #c = F.gumbel_softmax(c, tau=tau,hard=True, dim = -1)
    # Compute MinCut loss.
    out_adj = torch.matmul(torch.matmul(c.transpose(0, 1), adj), c)
    mincut_num = torch.einsum('ii->', out_adj)
    d_flat = adj.sum(dim=1)
    mincut_den = torch.einsum('ii->', torch.matmul(c.transpose(0, 1), d_flat.unsqueeze(-1) * c))
    mincut_loss = -(mincut_num / (mincut_den + 1e-15))
    mincut_loss = mincut_loss.mean()
    # Compute Orthogonality loss.
    ss = torch.matmul(c.transpose(0, 1), c)
    i_s = torch.eye(k, device=ss.device, dtype=ss.dtype)
    ortho_loss = torch.norm(
        ss / ss.norm(dim=(-1, -2), keepdim=True) -
        i_s / i_s.norm(), dim=(-1, -2))
    ortho_loss = ortho_loss.mean()
    return mincut_loss, ortho_loss

def structure_loss(z, edge_index, neigh_num):
    # edge_index[0] contains the indices of the source nodes (S)
    # edge_index[1] contains the indices of the target nodes (R)
    S_emb = z[edge_index[0]]
    R_emb = z[edge_index[1]]
    # Calculate the dot product between source and target embeddings
    dot_product = torch.sum(S_emb * R_emb, dim=-1)
    # Calculate the structure loss as per the TensorFlow code
    structure_loss = -torch.log(torch.sigmoid(dot_product) + 1e-15)
    # Reduce the loss by summing over all edges
    structure_loss = torch.mean(structure_loss) * neigh_num
    return structure_loss

def normalize_adj(adj):
    # Compute the degree matrix
    deg = adj.sum(dim=1)
    
    # Compute D^-1/2
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0  # Handle division by zero
    
    # Create D^-1/2 matrix
    d_mat_inv_sqrt = torch.diag(deg_inv_sqrt)
    
    # Compute the normalized adjacency matrix
    normalized_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    
    return normalized_adj

def update_adj(z):
    inner = torch.matmul(z, z.t())
    inner = torch.where(inner >= 0.7, inner, torch.tensor(0.0, device='cuda'))
    inner=normalize_adj(inner)
    return inner

# Define the pretrain loss function
def pretrain_loss_function(data, recon, z, adj, c, edge_index, n_neighbors):
    recon_loss = F.mse_loss(recon, data)
    struc_loss = structure_loss(z, edge_index, n_neighbors)
    mcl, ol = mincut_loss(adj, c)

    cluster_loss = mcl + ol + 1
    GATE_loss = recon_loss + struc_loss

    return GATE_loss, cluster_loss

def temporal_smoothness_loss2(z, t, num_neighbors):
    """
    Compute the temporal smoothness loss using a kernel-based approach inspired by t-SNE.
    
    Parameters:
    z (torch.Tensor): Latent embeddings of shape [num_cells, latent_dim]
    t (torch.Tensor): Time points of shape [num_cells, 1]
    num_neighbors (int): Number of nearest neighbors to consider
    
    Returns:
    torch.Tensor: Temporal smoothness loss
    """
    # Compute pairwise distances in latent space
    dist_matrix = torch.cdist(z, z, p=2)  # Shape: [num_cells, num_cells]

    # Find the indices of the nearest num_neighbors neighbors for each cell
    _, nn_indices = torch.topk(dist_matrix, k=num_neighbors+1, largest=False)  # Include self in neighbors

    # Exclude the self-neighbors (first index) and get the neighbor indices
    nn_indices = nn_indices[:, 1:]

    # Gather the distances of the nearest neighbors
    dist_neighbors = dist_matrix.gather(1, nn_indices)
    
    # Compute similarities using a Gaussian kernel (high-dimensional space)
    sigma = torch.mean(dist_neighbors) / 2
    sim_high = torch.exp(-dist_neighbors ** 2 / (2 * sigma ** 2))
    
    # Compute pairwise differences in time points
    t_diff_matrix = torch.cdist(t, t, p=1)  # Shape: [num_cells, num_cells]
    
    # Gather the time differences of the nearest neighbors
    t_diff_neighbors = t_diff_matrix.gather(1, nn_indices)
    
    # Compute similarities using a t-distribution (low-dimensional space)
    sim_low = 1 / (1 + t_diff_neighbors ** 2)
    
    # Normalize the similarities to create probability distributions
    sim_high = sim_high / (sim_high.sum(dim=1, keepdim=True) + 1e-8)
    sim_low = sim_low / (sim_low.sum(dim=1, keepdim=True) + 1e-8)

    # Compute the Kullback-Leibler divergence between the high and low similarity distributions
    kl_div = sim_high * (torch.log(sim_high + 1e-8) - torch.log(sim_low + 1e-8))
    
    # Average the KL divergence over the neighbors for each cell
    kl_div_mean_neighbors = kl_div.mean(dim=1)
    
    # Average the KL divergence over all cells
    loss = kl_div_mean_neighbors.mean()

    return loss

def pearson_loss(u_cluster, t_cluster, epsilon=1e-8):
    # Mean normalization
    u_mean = u_cluster - u_cluster.mean(dim=0, keepdim=True)
    t_mean = t_cluster - t_cluster.mean(dim=0, keepdim=True)
    
    # Compute covariance and variance directly
    covariance = (u_mean * t_mean).sum(dim=0)
    u_variance = (u_mean ** 2).sum(dim=0) 
    t_variance = (t_mean ** 2).sum(dim=0)
    
    # Compute the Pearson correlation for each gene without using sqrt
    pearson_correlation = covariance / (torch.sqrt(u_variance * t_variance + epsilon))
    
    return pearson_correlation

def correlation_loss(t, type_features, unspliced, cluster_matrix):
    num_cells, num_genes = type_features.shape
    total_loss = 0.0

    epsilon = 1e-8  # Small epsilon value to avoid division by zero

    # Ensure type_features is a floating point tensor
    type_features = type_features.float()

    # Determine the cluster assignment for each cell
    cluster_assignments = cluster_matrix.argmax(dim=1)

    # Expand t to match the shape of type_features
    t_expanded = t.expand(-1, num_genes)

    num_clusters = cluster_matrix.shape[1]  # Assuming cluster assignments are 0-indexed

    for cluster in range(num_clusters):
        cluster_mask = (cluster_assignments == cluster)
        num_cells_in_cluster = cluster_mask.sum().item()

        if num_cells_in_cluster < 2:  # Skip clusters with fewer than 2 cells
            continue

        # Extract the regulation values for the current cluster
        regulation_values = type_features[cluster_mask, :]

        t_cluster = t_expanded[cluster_mask, :]
        u_cluster = unspliced[cluster_mask, :]

        # Compute cosine similarity for each valid gene
        cos_sim_up = pearson_loss(t_cluster, u_cluster)
        # Apply weights
        reg_up_values = torch.mean(regulation_values,dim = 0,keepdim=True) # apply mean to cellls within each cluster
        # total_loss += 1 - torch.dot(F.normalize(reg_up_values,p=2,dim=1).view(-1), cos_sim_up)#/torch.sqrt(torch.tensor(type_features.shape[1])))
        # For DG1, we set:
        total_loss += 1 - torch.dot(F.normalize(reg_up_values,p=2,dim=1).view(-1), cos_sim_up)/torch.sqrt(torch.tensor(type_features.shape[1]))
        # # Apply weights
        # reg_up_values = regulation_values.mean(dim=0)
        # cos_sim_up_modified = torch.where(reg_up_values >= 0, 1 - cos_sim_up, 1 + cos_sim_up)
        # weighted_corr_loss_up = cos_sim_up_modified * reg_up_values.abs()
        # total_loss += torch.sum(weighted_corr_loss_up)
        # total_loss += torch.dot(F.normalize(reg_up_values,p=2,dim=1).view(-1), torch.where(reg_up_values >= 0, 1 - cos_sim_up, -1 - cos_sim_up))

    total_loss /= num_clusters
    return total_loss#  * 10

def correlation_loss_vec(t, type_features, unspliced, cluster_vector):
    num_cells, num_genes = type_features.shape
    total_loss = 0.0

    epsilon = 1e-8  # Small epsilon value to avoid division by zero

    # Ensure type_features is a floating point tensor
    type_features = type_features.float()

    # Determine the number of clusters from the cluster_vector
    # num_clusters = torch.unique(cluster_vector).numel()
    unique_clusters = torch.unique(cluster_vector)

    # Expand t to match the shape of type_features
    t_expanded = t.expand(-1, num_genes)
    valid_clusters = 0
    # for cluster in range(num_clusters):
    for cluster in unique_clusters: # 直接遍历存在的 ID
        cluster_mask = (cluster_vector == cluster)
        num_cells_in_cluster = cluster_mask.sum().item()

        if num_cells_in_cluster < 2:  # Skip clusters with fewer than 2 cells
            continue
        valid_clusters += 1
        # Extract the regulation values for the current cluster
        regulation_values = type_features[cluster_mask, :]

        t_cluster = t_expanded[cluster_mask, :]
        u_cluster = unspliced[cluster_mask, :]

        # Compute cosine similarity for each valid gene
        # cos_sim_up = F.cosine_similarity(t_cluster, u_cluster, dim=0, eps=epsilon)
        cos_sim_up = pearson_loss(t_cluster, u_cluster)
        
        # Apply weights
        reg_up_values = torch.mean(regulation_values,dim = 0,keepdim=True) # apply mean to cellls within each cluster
        # total_loss += 1 - torch.dot(F.normalize(reg_up_values,p=2,dim=1).view(-1), cos_sim_up)#/torch.sqrt(torch.tensor(type_features.shape[1])))
        # For DG1, we set:
        total_loss += 1 - torch.dot(F.normalize(reg_up_values,p=2,dim=1).view(-1), cos_sim_up)/torch.sqrt(torch.tensor(type_features.shape[1]))
        # total_loss += 1 - torch.dot(reg_up_values.sign() * F.softmax(reg_up_values.abs()), cos_sim_up)
        # total_loss += 1 - torch.dot(F.normalize(reg_up_values,p=1,dim=1).view(-1), cos_sim_up) # torch.dot(F.normalize(reg_up_values,p=1,dim=1).view(-1), cos_sim_up/torch.sqrt(torch.tensor(type_features.shape[1])))
        # cos_sim_up_modified = torch.where(reg_up_values >= 0, 1 - cos_sim_up, 1 + cos_sim_up)
        # total_loss += torch.dot(F.normalize(reg_up_values,p=2,dim=1).view(-1), torch.where(reg_up_values >= 0, 1 - cos_sim_up, -1 - cos_sim_up))
        # weighted_corr_loss_up = cos_sim_up_modified * reg_up_values.abs()
        # total_loss += torch.sum(weighted_corr_loss_up) # apply mean along genes

    if valid_clusters > 0:
        total_loss = total_loss / valid_clusters
    else:
        total_loss = torch.tensor(0.0, device=type_features.device)
    return total_loss


def get_mask_t_share(t, pyg_data):

    # Extract source and target nodes
    source_nodes = pyg_data.edge_index[0]  # shape: [num_edges]
    target_nodes = pyg_data.edge_index[1]  # shape: [num_edges]

    # Extract the corresponding time points for source and target nodes
    source_times = t[source_nodes]  # shape: [num_edges, 1]
    target_times = t[target_nodes]  # shape: [num_edges, 1]

    # Create the mask
    mask = target_times > source_times

    # Update the edge_index in the PyG data object
    pyg_data.time_edge_index = pyg_data.edge_index[:, mask.squeeze()]
    return pyg_data

def get_mask_t_gene(t, gene_adj):
    device = t.device
    t = t.to(gene_adj[0].device)
    used_adj = []
    for i in range(len(gene_adj)):
        gene_adj[i] = gene_adj[i]
        # Extract source and target nodes
        source_nodes = gene_adj[i][0]  # shape: [num_edges]
        target_nodes = gene_adj[i][1]  # shape: [num_edges]

        # Extract the corresponding time points for source and target nodes
        source_times = t[source_nodes]  # shape: [num_edges, 1]
        target_times = t[target_nodes]  # shape: [num_edges, 1]

        # Create the mask
        mask = target_times > source_times
        # Update the edge_index in used_adj
        used_adj.append(gene_adj[i][:, mask.squeeze()])
    t = t.to(device)
    return used_adj

def masked_max_aggregation(cosine_similarity, inverse_indices, k = 3):
    num_groups = inverse_indices.max().item() + 1
    num_features = cosine_similarity.size(1)
    
    # Initialize the mask tensor and the results tensor
    mask = torch.ones_like(cosine_similarity, dtype=torch.bool)
    results = torch.full((num_groups, k, num_features), float(-2), device=cosine_similarity.device)
    
    for i in range(int(k)):
        # Apply the current mask
        masked_cosine_similarity = cosine_similarity * mask.float() + (~mask).float() * float(-2)
        
        # Perform the max aggregation
        max_values, max_indices = scatter_max(masked_cosine_similarity, inverse_indices, dim=0)
        
        # Store the current max values in the results tensor
        results[:, i, :] = max_values
        
        col_indices = torch.arange(max_indices.size(1), device=cosine_similarity.device).unsqueeze(0).expand_as(max_indices)

        # Use advanced indexing to replace the values
        mask[max_indices, col_indices] = False
    return torch.mean(1 - torch.masked_select(results, results.ne(-2)))

def velocity_loss_share_single_top3_polish(pyg_data, up_type_features, cp, warm_up, k=3, batch_size=None):
    # 1. 预计算全局变量 (这些占用显存小，仅 [N_cells, N_genes]，保留在全局)
    cells, genes = pyg_data.unsplice.shape
    cp_reshaped = cp.view(cells, genes, 3)
    alpha, beta, gamma = cp_reshaped[..., 0], cp_reshaped[..., 1], cp_reshaped[..., 2]

    if warm_up:
        alpha = torch.where(up_type_features > 0, 
                          torch.tensor(1.0, device=alpha.device), 
                          torch.tensor(0.0, device=alpha.device))
    
    # 提前算出所有细胞的预测值 (Small Tensor: ~150MB)
    pred_vu_all = alpha - beta * pyg_data.unsplice
    pred_vs_all = beta * pyg_data.unsplice - gamma * pyg_data.splice
    
    # 获取全图的边索引
    src_all = pyg_data.time_edge_index[0]
    tgt_all = pyg_data.time_edge_index[1]
    
    # === 分支判断：如果不分 Batch (默认情况) ===
    if batch_size is None:
        # 原始逻辑：一次性计算所有边
        # Unspliced Diff
        unsplice_diff = pyg_data.unsplice[tgt_all] - pyg_data.unsplice[src_all]
        pred_vu_edge = pred_vu_all[src_all]
        
        # Spliced Diff
        splice_diff = pyg_data.splice[tgt_all] - pyg_data.splice[src_all]
        pred_vs_edge = pred_vs_all[src_all]

        # Calculate Cosine Similarity (All at once)
        cosine_sim = F.cosine_similarity(
            torch.stack((pred_vu_edge, pred_vs_edge), dim=2), 
            torch.stack((unsplice_diff, splice_diff), dim=2), 
            dim=2
        )
        
        _, inverse_indices = src_all.unique(return_inverse=True)
        # 直接聚合
        final_loss = masked_max_aggregation(cosine_sim, inverse_indices, k=k)
        return final_loss

    else:
        # === 分支判断：开启 Batch (大显存救急模式) ===
        # 获取所有作为 source 的唯一细胞 (用于分批)
        unique_src_cells, inverse_indices_all = src_all.unique(return_inverse=True)
        num_unique_cells = unique_src_cells.size(0)

        total_loss_sum = 0.0
        
        for i in range(0, num_unique_cells, batch_size):
            # 1. 确定当前批次的细胞
            batch_cell_indices = unique_src_cells[i : i + batch_size]
            
            # 2. 找到这些细胞发出的所有边
            edge_mask = torch.isin(src_all, batch_cell_indices)
            if not edge_mask.any(): continue

            # 3. 提取当前批次的边数据
            batch_src = src_all[edge_mask]
            batch_tgt = tgt_all[edge_mask]
            
            # 4. 准备计算
            unsplice_diff = pyg_data.unsplice[batch_tgt] - pyg_data.unsplice[batch_src]
            pred_vu_edge = pred_vu_all[batch_src]
            
            splice_diff = pyg_data.splice[batch_tgt] - pyg_data.splice[batch_src]
            pred_vs_edge = pred_vs_all[batch_src]

            # 5. 计算 Cosine
            cosine_sim = F.cosine_similarity(
                torch.stack((pred_vu_edge, pred_vs_edge), dim=2), 
                torch.stack((unsplice_diff, splice_diff), dim=2), 
                dim=2
            )
            
            # 6. 聚合 Loss
            _, batch_inverse = batch_src.unique(return_inverse=True)
            batch_mean_loss = masked_max_aggregation(cosine_sim, batch_inverse, k=k)
            
            # 7. 累加权重 Loss (还原回 sum)
            n_cells_in_batch = batch_inverse.max().item() + 1
            total_loss_sum += batch_mean_loss * n_cells_in_batch

        final_loss = total_loss_sum / num_unique_cells
        return final_loss

def velocity_loss_gene_single_top3_polish(device, device2, used_adj, pyg_data, up_type_features, cp, warm_up, k = 3):

    up_type_features = up_type_features.to(device2)
    cp = cp.to(device2)
    pyg_data.unsplice = pyg_data.unsplice.to(device2)
    pyg_data.splice = pyg_data.splice.to(device2)
    # Get the used material
    cells, genes = pyg_data.unsplice.shape
    # Reshape cp to separate alpha, beta, gamma for each gene
    cp_reshaped = cp.view(cells, genes, 3)
    alpha, beta, gamma = cp_reshaped[..., 0], cp_reshaped[..., 1], cp_reshaped[..., 2]

    # warm up
    if warm_up:
        alpha = torch.where(up_type_features > 0, torch.tensor(1.0, device=alpha.device), torch.tensor(0.0, device=alpha.device))
    
    # Calculate pred_vu and pred_vs for each gene, each cell
    pred_vu = alpha - beta * pyg_data.unsplice
    pred_vs = beta * pyg_data.unsplice - gamma * pyg_data.splice

    total_velo_loss = []
    for i in range(len(used_adj)):
        # Source and target nodes
        src = used_adj[i][0]
        tgt = used_adj[i][1]
        
        # Calculate differences for unsplice
        unsplice_diff = pyg_data.unsplice[tgt, i] - pyg_data.unsplice[src, i]
        pred_vu_time_edge = pred_vu[src, i]
        
        # Calculate differences for splice
        splice_diff = pyg_data.splice[tgt, i] - pyg_data.splice[src, i]
        pred_vs_time_edge = pred_vs[src, i]

        # Calculate cosine similarity
        cosine_similarity = F.cosine_similarity(torch.stack((pred_vu_time_edge, pred_vs_time_edge), dim=-1), torch.stack((unsplice_diff, splice_diff), dim=-1), dim=-1)
        
        _, inverse_indices = src.unique(return_inverse=True)

        velo_loss = masked_max_aggregation(cosine_similarity.unsqueeze(1), inverse_indices, k = k)
        total_velo_loss.append(velo_loss)

    mean_velo_loss = torch.mean(torch.stack(total_velo_loss))
    
    up_type_features = up_type_features.to(device)
    cp = cp.to(device)
    pyg_data.unsplice = pyg_data.unsplice.to(device)
    pyg_data.splice = pyg_data.splice.to(device)
    mean_velo_loss = mean_velo_loss.to(device)

    return mean_velo_loss

def velocity_loss_share_single_top3_polish_multi(device2, pyg_data, up_type_features, cp, warm_up, k = 3):
    device = cp.device
    # Get the used material
    cells, genes = pyg_data.unsplice.shape
    # Reshape cp to separate alpha, beta, gamma for each gene
    cp_reshaped = cp.view(cells, genes, 3)
    alpha, beta, gamma = cp_reshaped[..., 0], cp_reshaped[..., 1], cp_reshaped[..., 2]

    # warm up
    if warm_up:
        alpha = torch.where(up_type_features > 0, torch.tensor(1.0, device=alpha.device), torch.tensor(0.0, device=alpha.device))
    
    # Calculate pred_vu and pred_vs for each gene, each cell
    pred_vu = alpha - beta * pyg_data.unsplice
    pred_vs = beta * pyg_data.unsplice - gamma * pyg_data.splice

    # Source and target nodes
    src = pyg_data.time_edge_index[0]
    tgt = pyg_data.time_edge_index[1]
    
    # Calculate differences for unsplice
    unsplice_diff = pyg_data.unsplice[tgt] - pyg_data.unsplice[src]
    pred_vu_time_edge = pred_vu[src]
    
    # Calculate differences for splice
    splice_diff = pyg_data.splice[tgt] - pyg_data.splice[src]
    pred_vs_time_edge = pred_vs[src]

    pred_vu_time_edge = pred_vu_time_edge.to(device2)
    pred_vs_time_edge = pred_vs_time_edge.to(device2)
    unsplice_diff = unsplice_diff.to(device2)
    splice_diff = splice_diff.to(device2)

    # Calculate cosine similarity
    cosine_similarity = F.cosine_similarity(torch.stack((pred_vu_time_edge, pred_vs_time_edge), dim=2), torch.stack((unsplice_diff, splice_diff), dim=2), dim=2)
    
    _, inverse_indices = src.unique(return_inverse=True)

    inverse_indices = inverse_indices.to(device2)

    velo_loss = masked_max_aggregation(cosine_similarity, inverse_indices, k = k)
    
    # Move tensors back to the original device
    velo_loss = velo_loss.to(device)

    return velo_loss

# def post_loss_share(device2, z, pyg_data, cp, up_type_features, t, c_softmax, n_neighbors, warm_up, time_only = False, corr_mode = 'u', cos_batch = 500):

#     if time_only:
#         # Velo loss
#         velo_loss = 0
#         # Time cor loss
#         time_cor = correlation_loss_vec(t, up_type_features, pyg_data.unsplice, pyg_data.pre_clus_vec)
#     else:
#         # velo loss
#         pyg_data = get_mask_t_share(t, pyg_data)
#         if device2 != t.device:
#             velo_loss = velocity_loss_share_single_top3_polish_multi(device2, pyg_data, up_type_features, cp, warm_up)
#         else:
#             velo_loss = velocity_loss_share_single_top3_polish(pyg_data, up_type_features, cp, warm_up)
#         # cor constrain loss
#         # cons_loss = F.mse_loss(up_type_features, pyg_data.type_features/up_type_features.shape[1])
#         # Time cor loss
#         if corr_mode == 'u':
#             time_cor = correlation_loss(t, up_type_features, pyg_data.unsplice, c_softmax)
#             time_cor = time_cor# + cons_loss
#         elif corr_mode == 's':
#             time_cor = correlation_loss(t, up_type_features, pyg_data.splice, c_softmax)
#             time_cor = time_cor# + cons_loss
#         elif corr_mode == 'all':
#             time_cor = 0.5 * correlation_loss(t, up_type_features, pyg_data.unsplice, c_softmax) + 0.5 * correlation_loss(t, up_type_features, pyg_data.splice, c_softmax)
#             time_cor = time_cor# + cons_loss
#     # Time Smooth loss
#     temp_smooth_loss = temporal_smoothness_loss2(z, t, n_neighbors)

#     return velo_loss, time_cor, temp_smooth_loss

# 修改后 (增加 fine_cluster_vec 参数)
def post_loss_share(device2, z, pyg_data, cp, up_type_features, t, c_softmax, n_neighbors, warm_up, time_only=False, corr_mode='u', cos_batch=500, fine_cluster_vec=None, velo_batch_size=None):
    # 1. 处理 Time Correlation Loss
    # 逻辑：只要有细聚类 (fine_cluster_vec)，就用细类算 Correlation；否则用 Experts (c_softmax)
    if fine_cluster_vec is not None:
        # === 方案 A: 使用细聚类 (硬分类) ===
        if corr_mode == 'u':
            time_cor = correlation_loss_vec(t, up_type_features, pyg_data.unsplice, fine_cluster_vec)
        elif corr_mode == 's':
            time_cor = correlation_loss_vec(t, up_type_features, pyg_data.splice, fine_cluster_vec)
        elif corr_mode == 'all':
            time_cor = 0.5 * correlation_loss_vec(t, up_type_features, pyg_data.unsplice, fine_cluster_vec) + \
                       0.5 * correlation_loss_vec(t, up_type_features, pyg_data.splice, fine_cluster_vec)
    else:
        # === 方案 B: 使用 Experts (软分类 argmax) ===
        # 注意：Time Only 阶段原来用的是 pyg_data.pre_clus_vec，这里要做个兼容
        target_cluster = pyg_data.pre_clus_vec if time_only else c_softmax
        
        if time_only:
            # Time Only 阶段传进来的是向量 (pre_clus_vec)
            if corr_mode == 'u':
                time_cor = correlation_loss_vec(t, up_type_features, pyg_data.unsplice, target_cluster)
            elif corr_mode == 's':
                time_cor = correlation_loss_vec(t, up_type_features, pyg_data.splice, target_cluster)
            elif corr_mode == 'all':
                time_cor = 0.5 * correlation_loss_vec(t, up_type_features, pyg_data.unsplice, target_cluster) + \
                        0.5 * correlation_loss_vec(t, up_type_features, pyg_data.splice, target_cluster)
        else:
            # 正常训练阶段传进来的是矩阵 (c_softmax)
            if corr_mode == 'u':
                time_cor = correlation_loss(t, up_type_features, pyg_data.unsplice, target_cluster)
            elif corr_mode == 's':
                time_cor = correlation_loss(t, up_type_features, pyg_data.splice, target_cluster)
            elif corr_mode == 'all':
                time_cor = 0.5 * correlation_loss(t, up_type_features, pyg_data.unsplice, target_cluster) + \
                        0.5 * correlation_loss(t, up_type_features, pyg_data.splice, target_cluster)
    
    # 2. 处理 Velocity Loss
    if time_only:
        velo_loss = 0
    else:
        pyg_data = get_mask_t_share(t, pyg_data)
        if device2 != t.device:
            velo_loss = velocity_loss_share_single_top3_polish_multi(device2, pyg_data, up_type_features, cp, warm_up)
        else:
            velo_loss = velocity_loss_share_single_top3_polish(pyg_data, up_type_features, cp, warm_up, batch_size=velo_batch_size)

    # Time Smooth loss (不变)
    temp_smooth_loss = temporal_smoothness_loss2(z, t, n_neighbors)

    return velo_loss, time_cor, temp_smooth_loss

def prior_time_loss(t, prior_time):
    t = t.squeeze()  # Remove the extra dimension from t, so it's of shape (9815,)
    # Get unique cluster labels and their inverse indices
    unique_clusters, inverse_indices = torch.unique(prior_time, return_inverse=True)

    # Calculate the sum and count of predicted times for each cluster using scatter_add
    cluster_sums = torch.zeros_like(unique_clusters, dtype=torch.float).scatter_add_(0, inverse_indices, t)
    cluster_counts = torch.zeros_like(unique_clusters, dtype=torch.float).scatter_add_(0, inverse_indices, torch.ones_like(t))

    # Compute the average predicted time for each cluster
    cluster_pred_times = cluster_sums / cluster_counts

    # The cluster_prior_times are simply the unique cluster labels
    cluster_prior_times = unique_clusters.float()

    # Mean of cluster_pred_times and cluster_prior_times
    mean_pred = torch.mean(cluster_pred_times)
    mean_prior = torch.mean(cluster_prior_times)

    # Compute covariance between cluster_pred_times and cluster_prior_times
    covariance = torch.mean((cluster_pred_times - mean_pred) * (cluster_prior_times - mean_prior))

    # Compute standard deviations of cluster_pred_times and cluster_prior_times
    std_pred = torch.std(cluster_pred_times)
    std_prior = torch.std(cluster_prior_times)

    # Compute the Pearson correlation coefficient at the cluster level
    correlation = covariance / (std_pred * std_prior)
    return 1 - correlation

# def post_loss_share_prior_time(device2, z, pyg_data, cp, up_type_features, t, c_softmax, n_neighbors, warm_up, time_only = False, corr_mode = 'u', prior_time = None):

#     # Time loss
#     if corr_mode == 'u':
#         time_cor = correlation_loss(t, up_type_features, pyg_data.unsplice, c_softmax)
#     elif corr_mode == 's':
#         time_cor = correlation_loss(t, up_type_features, pyg_data.splice, c_softmax)
#     elif corr_mode == 'all':
#         time_cor = 0.5 * correlation_loss(t, up_type_features, pyg_data.unsplice, c_softmax) + 0.5 * correlation_loss(t, up_type_features, pyg_data.splice, c_softmax)
#     temp_smooth_loss = temporal_smoothness_loss2(z, t, n_neighbors)
    
#     if prior_time is not None:
#         time_cor = 0.5 * time_cor + 0.5 * prior_time_loss(t, prior_time)

#     if time_only:
#         velo_loss = 0
#     else:
#         pyg_data = get_mask_t_share(t, pyg_data)
#         if device2 != time_cor.device:
#             velo_loss = velocity_loss_share_single_top3_polish_multi(device2, pyg_data, up_type_features, cp, warm_up)
#         else:
#             velo_loss = velocity_loss_share_single_top3_polish(pyg_data, up_type_features, cp, warm_up)

#     return velo_loss, time_cor, temp_smooth_loss


def post_loss_gene(gene_adj, device, device2, z, pyg_data, cp, up_type_features, t, c_softmax, n_neighbors, warm_up, time_only = False, corr_mode = 'u', cos_batch = 500):

    # Time loss
    if corr_mode == 'u':
        time_cor = correlation_loss(t, up_type_features, pyg_data.unsplice, c_softmax)
    elif corr_mode == 's':
        time_cor = correlation_loss(t, up_type_features, pyg_data.splice, c_softmax)
    elif corr_mode == 'all':
        time_cor = 0.5 * correlation_loss(t, up_type_features, pyg_data.unsplice, c_softmax) + 0.5 * correlation_loss(t, up_type_features, pyg_data.splice, c_softmax)
    temp_smooth_loss = temporal_smoothness_loss2(z, t, n_neighbors)

    if time_only:
        velo_loss = 0
    else:
        used_adj = get_mask_t_gene(t, gene_adj)
        velo_loss = velocity_loss_gene_single_top3_polish(device, device2,used_adj, pyg_data, up_type_features, cp, warm_up)

    return velo_loss, time_cor, temp_smooth_loss

def generate_mask(x, mask_ratio=0.1):
    mask = torch.rand_like(x) < mask_ratio
    return mask.float()

# Function to train and evaluate for each gene
def model_training_share_neighbor_adata(device, device2, pyg_data, MODEL_MODE, adata, 
                                        NUM_LOSS_NEIGH, max_n_cluster, corr_mode = 'u', 
                                        cos_batch = 100, path = '/HOME/scz3472/run/GATVelo/', mask_train = False, prior_time = None, order = None,
                                        hidden_size = 512, latent_size = 128, num_epochs = 10000,pretrain_epochs = 1500,cluster_epochs = 1500,time_epochs = 200,warm_up_epochs = 200,expert_mode='original', velo_batch_size=None):
    if MODEL_MODE == 'pretrain':
        use_experts = False
    else:
        use_experts = True

    if prior_time is not None:
        adata.obs[prior_time] = pd.Categorical(adata.obs[prior_time], categories=order, ordered=True)
        # Convert to PyTorch tensor if necessary
        numeric_stage_tensor = torch.tensor(adata.obs[prior_time].cat.codes.values, dtype=torch.float32, device = device)
    # Set random seed inside the function
    SEED = 618
    torch.manual_seed(SEED)
    # assign the former cluster
    if MODEL_MODE != 'pretrain':
        pyg_data.pre_clus_vec = torch.tensor(adata.obs['pred_cluster'].values, dtype=torch.long, device = device)
    # Prepare data
    pyg_data = pyg_data.to(device)
    # 【优化】在循环外提取细聚类向量 (Fine Cluster Vector)
    # 这样避免了在每个 epoch 里重复查找属性
    fine_clus_vec = getattr(pyg_data, 'fine_clus_vec', None)
    if fine_clus_vec is not None:
        print("Using Fine Cluster Vector for Correlation Loss.")
    # Initialize the GATE model
    model = GATModel(
        in_features=pyg_data.x.shape[1],
        hid_features=hidden_size,
        out_features=latent_size,
        num_clusters=max_n_cluster,
        num_genes=adata.n_vars,
        type_features=pyg_data.type_features,
        use_experts=use_experts,
        expert_mode=expert_mode # <--- 传入 GATModel
    ).to(device)
    pyg_data.unsplice = pyg_data.x[:,:adata.n_vars]
    pyg_data.splice = pyg_data.x[:,adata.n_vars:]

    # Define optimizers
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-5
    )
    # Early stopping parameters
    early_stopping_patience = 1000 # 1000 before
    min_loss_improvement = 0.001
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None  # 用来暂存最佳参数
    # Training loop
    # num_epochs = 10000
    # pretrain_epochs = 1500
    # cluster_epochs = 1500
    # time_epochs = 200 # 1000 before
    # warm_up_epochs = 200 #1000 before
    log_interval = 50
    # update_interval = 50 # Update adj matrix for mincut loss
    inner_adj = None
    freeze_done = False
    model.train()

    if MODEL_MODE == 'pretrain':
        MAX_EPOCH = pretrain_epochs + cluster_epochs
    else:
        MAX_EPOCH = num_epochs
    
    for epoch in range(MAX_EPOCH):
        optimizer.zero_grad()
        
        if epoch < pretrain_epochs:
            # Phase 1: Just train the GATE
            if mask_train:
                mask = generate_mask(pyg_data.x, 0.1)
                x_unmasked = pyg_data.x * (1 - mask)
                recon, z, t, c_softmax, cp, up_type_features = model(x_unmasked, pyg_data.edge_index, use_initial_regulation=True)
                GATE_loss, _ = pretrain_loss_function(data = pyg_data.x, recon = recon, z = z, adj = pyg_data.adj, c = c_softmax, edge_index = pyg_data.edge_index, n_neighbors = NUM_LOSS_NEIGH)
            else:
                recon, z, t, c_softmax, cp, up_type_features = model(pyg_data.x, pyg_data.edge_index, use_initial_regulation=True)
                GATE_loss, _ = pretrain_loss_function(data = pyg_data.x, recon = recon, z = z, adj = pyg_data.adj, c = c_softmax, edge_index = pyg_data.edge_index, n_neighbors = NUM_LOSS_NEIGH)
            loss = GATE_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
        elif epoch <= pretrain_epochs + cluster_epochs:
            # Phase 2: Train the GATE and MinCUT
            if mask_train:
                mask = generate_mask(pyg_data.x, 0.1)
                x_unmasked = pyg_data.x * (1 - mask)
                recon, z, t, c_softmax, cp, up_type_features = model(x_unmasked, pyg_data.edge_index, use_initial_regulation=True)
                GATE_loss, cluster_loss = pretrain_loss_function(data = pyg_data.x, recon = recon, z = z, adj = pyg_data.adj, c = c_softmax, edge_index = pyg_data.edge_index, n_neighbors = NUM_LOSS_NEIGH)
            else:
                recon, z, t, c_softmax, cp, up_type_features = model(pyg_data.x, pyg_data.edge_index, use_initial_regulation=True)
                GATE_loss, cluster_loss = pretrain_loss_function(data = pyg_data.x, recon = recon, z = z, adj = pyg_data.adj, c = c_softmax, edge_index = pyg_data.edge_index, n_neighbors = NUM_LOSS_NEIGH)
            loss = GATE_loss + cluster_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
        elif epoch <= pretrain_epochs + cluster_epochs + time_epochs:
            # Phase 3: Train the time
            if mask_train:
                mask = generate_mask(pyg_data.x, 0.1)
                x_unmasked = pyg_data.x * (1 - mask)
                recon, z, t, c_softmax, cp, up_type_features = model(x_unmasked, pyg_data.edge_index, use_initial_regulation=True)
                GATE_loss, cluster_loss = pretrain_loss_function(data = pyg_data.x, recon = recon, z = z, adj = pyg_data.adj, c = c_softmax, edge_index = pyg_data.edge_index, n_neighbors = NUM_LOSS_NEIGH)
            else:
                recon, z, t, c_softmax, cp, up_type_features = model(pyg_data.x, pyg_data.edge_index, use_initial_regulation=True)
                GATE_loss, cluster_loss = pretrain_loss_function(data = pyg_data.x, recon = recon, z = z, adj = pyg_data.adj, c = c_softmax, edge_index = pyg_data.edge_index, n_neighbors = NUM_LOSS_NEIGH)
            
            _, time_cor, temp_smooth_loss = post_loss_share(
                device2, z, pyg_data, cp, up_type_features, t, c_softmax, 
                n_neighbors=NUM_LOSS_NEIGH, warm_up=True, time_only=True, 
                corr_mode=corr_mode, 
                fine_cluster_vec=fine_clus_vec  # <--- Use Fine Cluster
            )
            loss = GATE_loss + cluster_loss + time_cor + temp_smooth_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
        elif epoch <= pretrain_epochs + cluster_epochs + time_epochs + warm_up_epochs:
            # Phase 4: Train the velocity(warm up)
            if mask_train:
                mask = generate_mask(pyg_data.x, 0.1)
                x_unmasked = pyg_data.x * (1 - mask)
                recon, z, t, c_softmax, cp, up_type_features = model(x_unmasked, pyg_data.edge_index, use_initial_regulation=False)
                GATE_loss, cluster_loss = pretrain_loss_function(data = pyg_data.x, recon = recon, z = z, adj = pyg_data.adj, c = c_softmax, edge_index = pyg_data.edge_index, n_neighbors = NUM_LOSS_NEIGH)
            else:
                recon, z, t, c_softmax, cp, up_type_features = model(pyg_data.x, pyg_data.edge_index, use_initial_regulation=False)
                GATE_loss, cluster_loss = pretrain_loss_function(data = pyg_data.x, recon = recon, z = z, adj = pyg_data.adj, c = c_softmax, edge_index = pyg_data.edge_index, n_neighbors = NUM_LOSS_NEIGH)
            # 【修改】传入 fine_clus_vec
            velo_loss, time_cor, temp_smooth_loss = post_loss_share(
                device2, z, pyg_data, cp, up_type_features, t, c_softmax, 
                n_neighbors=NUM_LOSS_NEIGH, warm_up=True, 
                corr_mode=corr_mode,
                fine_cluster_vec=fine_clus_vec, # <--- Use Fine Cluster
                velo_batch_size=velo_batch_size # <--- 传入
            )
            loss = GATE_loss + cluster_loss + velo_loss + time_cor + temp_smooth_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
        else:
            # Phase 5: Train the velocity
            if mask_train:
                mask = generate_mask(pyg_data.x, 0.1)
                x_unmasked = pyg_data.x * (1 - mask)
                recon, z, t, c_softmax, cp, up_type_features = model(x_unmasked, pyg_data.edge_index, use_initial_regulation=False)
                GATE_loss, cluster_loss = pretrain_loss_function(data = pyg_data.x, recon = recon, z = z, adj = pyg_data.adj, c = c_softmax, edge_index = pyg_data.edge_index, n_neighbors = NUM_LOSS_NEIGH)
            else:
                recon, z, t, c_softmax, cp, up_type_features = model(pyg_data.x, pyg_data.edge_index, use_initial_regulation=False)
                GATE_loss, cluster_loss = pretrain_loss_function(data = pyg_data.x, recon = recon, z = z, adj = pyg_data.adj, c = c_softmax, edge_index = pyg_data.edge_index, n_neighbors = NUM_LOSS_NEIGH)
            velo_loss, time_cor, temp_smooth_loss = post_loss_share(
                device2, z, pyg_data, cp, up_type_features, t, c_softmax, 
                n_neighbors=NUM_LOSS_NEIGH, warm_up=False, 
                corr_mode=corr_mode,
                fine_cluster_vec=fine_clus_vec, # <--- Use Fine Cluster
                velo_batch_size=velo_batch_size # <--- 传入
            )
            loss = GATE_loss + cluster_loss + velo_loss + time_cor + temp_smooth_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            scheduler.step()
            current_loss = loss.item()
            if current_loss < best_loss - min_loss_improvement:
                best_loss = current_loss
                patience_counter = 0  # Reset patience counter on improvement
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1  # Increment patience counter if no improvement
            
            # Check for early stopping condition
            if patience_counter > early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}. Best loss was {best_loss}")
                break  # Exit the training loop

        if epoch % log_interval == 0:
            print(f"Epoch {epoch}, Loss {loss.item()}")
            if epoch > pretrain_epochs + cluster_epochs + time_epochs:
                print(f"GATE: {GATE_loss}, Cluster: {cluster_loss}, Velo: {velo_loss}, Time_cor: {time_cor}, Time_smooth: {temp_smooth_loss}")
            elif epoch > pretrain_epochs + cluster_epochs:
                print(f"GATE: {GATE_loss}, Cluster: {cluster_loss}, Time_cor: {time_cor}, Time_smooth: {temp_smooth_loss}")
    
    # === 训练结束后，一定要加载回来 ===
    if best_model_state is not None:
        print("Loading best model state from memory...")
        model.load_state_dict(best_model_state)
    # 此时 model 变回了 Loss 最低时的状态，用这个去 eval 才是最好的！
    
    # # Save the final model including the scheduler state
    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'scheduler_state_dict': scheduler.state_dict(),
    #     'loss': loss,
    # }, path + MODEL_MODE + '_GATEVelo_check_point.pth')

    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        num_genes = adata.n_vars
        recon, z, t, c_softmax, cp, up_type_features = model(pyg_data.x, pyg_data.edge_index, use_initial_regulation=False)
        # Identify the cluster with the highest probability for each cell
        cluster = torch.argmax(c_softmax, dim=1)
        weight = torch.max(c_softmax, dim=1).values

        # Reshape cp to (num_cells, num_genes, 3)
        cp_reshaped = cp.view(recon.shape[0], num_genes, 3)
        alpha, beta, gamma = cp_reshaped[..., 0], cp_reshaped[..., 1], cp_reshaped[..., 2]

        # Convert tensors to numpy for further processing/storage
        orig = pyg_data.x.cpu().numpy()
        recon = recon.cpu().numpy()
        cluster = cluster.cpu().numpy()
        weight = weight.cpu().numpy()
        # cell_ids = cell_ids.cpu().numpy()
        # Assuming cell_ids can be either a tensor or a numpy array
        cell_ids = pyg_data.cell_ids.cpu().numpy() if isinstance(pyg_data.cell_ids, torch.Tensor) else pyg_data.cell_ids

        z_numpy = z.cpu().numpy()
        t = t.cpu().numpy()
        up_type_features = up_type_features.cpu().numpy()

        if MODEL_MODE == 'pretrain':
            # Add the results
            adata.layers['recon_u'] = recon[:,:adata.n_vars]
            adata.layers['recon_s'] = recon[:,adata.n_vars:]
            adata.layers['scale_Mu'] = orig[:,:adata.n_vars]
            adata.layers['scale_Ms'] = orig[:,adata.n_vars:]
            adata.layers['pred_cell_type'] = up_type_features
            adata.obs['pred_cluster'] = cluster
            adata.obs['pred_clus_weight'] = weight
            adata.obsm['X_pre_embed'] = z_numpy
            adata.obsm['cluster_matrix'] = c_softmax.cpu().numpy()
            # adata.uns['coar_adj'] = coar_adj.detach().cpu().numpy()
            # Calculate neighbors based on 'X_pre_embed' and store them in a new slot
            sc.pp.neighbors(adata,n_neighbors=30, use_rep='X_pre_embed', key_added='pre_embed_neighbors')
            # Calculate UMAP based on the new neighbors and store it in a new slot
            temp_adata = sc.tl.umap(adata, neighbors_key='pre_embed_neighbors',random_state=618, copy = True)
            adata.obsm['X_umap_pre_embed'] = temp_adata.obsm['X_umap']
            del temp_adata
            adata.obs['pred_cluster'] = adata.obs['pred_cluster'].astype('category')
        else:
            # Add the results
            adata.layers['recon_u_refine'] = recon[:,:adata.n_vars]
            adata.layers['recon_s_refine'] = recon[:,adata.n_vars:]
            adata.layers['scale_Mu_refine'] = orig[:,:adata.n_vars]
            adata.layers['scale_Ms_refine'] = orig[:,adata.n_vars:]
            adata.layers['pred_cell_type_refine'] = up_type_features
            adata.layers['recon_alpha'], adata.layers['recon_beta'], adata.layers['recon_gamma'] = alpha.cpu().numpy(), beta.cpu().numpy(), gamma.cpu().numpy()
            adata.layers['pred_vu'] = adata.layers['recon_alpha'] - adata.layers['recon_beta'] * adata.layers['scale_Mu_refine']
            adata.layers['pred_vs'] = adata.layers['recon_beta'] * adata.layers['scale_Mu_refine'] - adata.layers['recon_gamma'] * adata.layers['scale_Ms_refine']
            adata.obs['pred_cluster_refine'] = cluster
            adata.obs['pred_clus_weight_refine'] = weight
            adata.obs['pred_time'] = t
            adata.obsm['X_refine_embed'] = z_numpy
            adata.obsm['cluster_matrix'] = c_softmax.cpu().numpy()
            # adata.uns['coar_adj'] = coar_adj.detach().cpu().numpy()
            # Calculate neighbors based on 'X_refine_embed' and store them in a new slot
            sc.pp.neighbors(adata,n_neighbors=30, use_rep='X_refine_embed', key_added='refine_embed_neighbors')
            # Calculate UMAP based on the new neighbors and store it in a new slot
            temp_adata = sc.tl.umap(adata, neighbors_key='refine_embed_neighbors',random_state=618, copy = True)
            adata.obsm['X_umap_refine_embed'] = temp_adata.obsm['X_umap']
            del temp_adata
            adata.obs['pred_cluster_refine'] = adata.obs['pred_cluster_refine'].astype('category')

    model.to('cpu')
    # Save only the model's state_dict
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path + MODEL_MODE + '_GATEVelo_check_point.pth')
    # Return the DataFrame for the current gene
    return adata

def inductive_learn(device, adata, pyg_data, max_n_cluster, path, hidden_size = 512, latent_size = 128):
    # Set random seed inside the function
    SEED = 618
    torch.manual_seed(SEED)
    # Prepare data
    pyg_data = pyg_data.to(device)
    pyg_data.x[torch.isnan(pyg_data.x)] = 0
    # Initialize the GATE model
    model = GATModel(
        in_features=pyg_data.x.shape[1],
        hid_features=hidden_size,
        out_features=latent_size,
        num_clusters=max_n_cluster,
        num_genes=adata.n_vars,
        type_features=pyg_data.type_features,
        use_experts=True
    ).to(device)
    pyg_data.unsplice = pyg_data.x[:,:adata.n_vars]
    pyg_data.splice = pyg_data.x[:,adata.n_vars:]

    checkpoint = torch.load(path + 'whole_GATEVelo_check_point.pth',map_location=device)
    # Filter out the 'up_regulation.type_features' parameter from the state dictionary
    state_dict = checkpoint['model_state_dict']
    state_dict = {k: v for k, v in state_dict.items() if 'up_regulation.type_features' not in k and 'up_regulation.learnable_matrix' not in k}

    model.load_state_dict(state_dict,strict=False)

    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        num_genes = adata.n_vars
        recon, z, t, c_softmax, cp, up_type_features = model(pyg_data.x, pyg_data.edge_index, use_initial_regulation=False)
        # Identify the cluster with the highest probability for each cell
        cluster = torch.argmax(c_softmax, dim=1)
        weight = torch.max(c_softmax, dim=1).values

        # Reshape cp to (num_cells, num_genes, 3)
        cp_reshaped = cp.view(recon.shape[0], num_genes, 3)
        alpha, beta, gamma = cp_reshaped[..., 0], cp_reshaped[..., 1], cp_reshaped[..., 2]

        # Convert tensors to numpy for further processing/storage
        orig = pyg_data.x.cpu().numpy()
        recon = recon.cpu().numpy()
        cluster = cluster.cpu().numpy()
        weight = weight.cpu().numpy()
        # cell_ids = cell_ids.cpu().numpy()
        # Assuming cell_ids can be either a tensor or a numpy array
        cell_ids = pyg_data.cell_ids.cpu().numpy() if isinstance(pyg_data.cell_ids, torch.Tensor) else pyg_data.cell_ids

        z_numpy = z.cpu().numpy()
        t = t.cpu().numpy()
        up_type_features = up_type_features.cpu().numpy()

        # Add the results
        adata.layers['recon_u_refine'] = recon[:,:adata.n_vars]
        adata.layers['recon_s_refine'] = recon[:,adata.n_vars:]
        adata.layers['scale_Mu_refine'] = orig[:,:adata.n_vars]
        adata.layers['scale_Ms_refine'] = orig[:,adata.n_vars:]
        # adata.layers['pred_cell_type_refine'] = up_type_features
        adata.layers['recon_alpha'], adata.layers['recon_beta'], adata.layers['recon_gamma'] = alpha.cpu().numpy(), beta.cpu().numpy(), gamma.cpu().numpy()
        adata.layers['pred_vu'] = adata.layers['recon_alpha'] - adata.layers['recon_beta'] * adata.layers['scale_Mu_refine']
        adata.layers['pred_vs'] = adata.layers['recon_beta'] * adata.layers['scale_Mu_refine'] - adata.layers['recon_gamma'] * adata.layers['scale_Ms_refine']
        adata.obs['pred_cluster_refine'] = cluster
        adata.obs['pred_clus_weight_refine'] = weight
        adata.obs['pred_time'] = t
        adata.obsm['X_refine_embed'] = z_numpy
        adata.obsm['cluster_matrix'] = c_softmax.cpu().numpy()
        # adata.uns['coar_adj'] = coar_adj.detach().cpu().numpy()
        # Calculate neighbors based on 'X_refine_embed' and store them in a new slot
        sc.pp.neighbors(adata,n_neighbors=30, use_rep='X_refine_embed', key_added='refine_embed_neighbors')
        # Calculate UMAP based on the new neighbors and store it in a new slot
        temp_adata = sc.tl.umap(adata, neighbors_key='refine_embed_neighbors', copy = True)
        adata.obsm['X_umap_refine_embed'] = temp_adata.obsm['X_umap']
        del temp_adata
        adata.obs['pred_cluster_refine'] = adata.obs['pred_cluster_refine'].astype('category')
    return adata

# Function to train and evaluate for each gene
def model_training_gene_neighbor_adata(gene_adj, device, device2, pyg_data, MODEL_MODE, adata, NUM_LOSS_NEIGH, max_n_cluster, FREEZE = False, corr_mode = 'u', cos_batch = 100):
    # Set random seed inside the function
    SEED = 618
    torch.manual_seed(SEED)
    # Prepare data
    pyg_data = pyg_data.to(device)
    # Initialize the GATE model
    model = GATModel(
        in_features=pyg_data.x.shape[1],
        hid_features=512,
        out_features=128,
        num_clusters=max_n_cluster,
        num_genes=adata.n_vars,
        type_features=pyg_data.type_features
    ).to(device)
    pyg_data.unsplice = pyg_data.x[:,:adata.n_vars]
    pyg_data.splice = pyg_data.x[:,adata.n_vars:]

    # Define optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    # Early stopping parameters
    early_stopping_patience = 1000
    min_loss_improvement = 0.001
    best_loss = float('inf')
    patience_counter = 0
    # Training loop
    num_epochs = 10000
    pretrain_epochs = 1500
    cluster_epochs = 1500
    time_epochs = 200 # 1000 before
    warm_up_epochs = 200 #1000 before
    log_interval = 50
    update_interval = 50 # Update adj matrix for mincut loss
    inner_adj = None
    freeze_done = False
    model.train()

    if MODEL_MODE == 'pretrain':
        MAX_EPOCH = pretrain_epochs + cluster_epochs
    else:
        MAX_EPOCH = num_epochs
    
    for epoch in range(MAX_EPOCH):
        optimizer.zero_grad()

        if epoch < pretrain_epochs:
            # Phase 1: Just train the GATE
            recon, z, t, c_softmax, cp, up_type_features = model(pyg_data.x, pyg_data.edge_index, use_initial_regulation=True)
            GATE_loss, _ = pretrain_loss_function(data = pyg_data.x, recon = recon, z = z, adj = pyg_data.adj, c = c_softmax, edge_index = pyg_data.edge_index, n_neighbors = NUM_LOSS_NEIGH)
            loss = GATE_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
        elif epoch <= pretrain_epochs + cluster_epochs:
            # Phase 2: Train the GATE and MinCUT
            recon, z, t, c_softmax, cp, up_type_features = model(pyg_data.x, pyg_data.edge_index, use_initial_regulation=True)
            #if epoch % update_interval == 0 or inner_adj is None:
            inner_adj = update_adj(z)      
            GATE_loss, cluster_loss = pretrain_loss_function(data = pyg_data.x, recon = recon, z = z, adj = inner_adj, c = c_softmax, edge_index = pyg_data.edge_index, n_neighbors = NUM_LOSS_NEIGH)
            loss = GATE_loss + cluster_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
        elif epoch <= pretrain_epochs + cluster_epochs + time_epochs:
            # Phase 3: Train the time
            recon, z, t, c_softmax, cp, up_type_features = model(pyg_data.x, pyg_data.edge_index, use_initial_regulation=True)
            #if epoch % update_interval == 0 or inner_adj is None:
            inner_adj = update_adj(z)       
            GATE_loss, cluster_loss = pretrain_loss_function(data = pyg_data.x, recon = recon, z = z, adj = inner_adj, c = c_softmax, edge_index = pyg_data.edge_index, n_neighbors = NUM_LOSS_NEIGH)
            _, time_cor, temp_smooth_loss = post_loss_gene(gene_adj, device, device2, z, pyg_data, cp, up_type_features, t, c_softmax, n_neighbors=NUM_LOSS_NEIGH, warm_up = True, time_only = True, corr_mode = corr_mode, cos_batch = cos_batch)
            loss = GATE_loss + cluster_loss + time_cor + temp_smooth_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
        elif epoch <= pretrain_epochs + cluster_epochs + time_epochs + warm_up_epochs:
            # freeze paramters
            if FREEZE and not freeze_done:
                for name, param in model.named_parameters():
                    if any(x in name for x in ['gat_conv', 'batch_norm']):
                        param.requires_grad = False
                freeze_done = True
            # Phase 4: Train the velocity(warm up)
            recon, z, t, c_softmax, cp, up_type_features = model(pyg_data.x, pyg_data.edge_index, use_initial_regulation=True)
            #if epoch % update_interval == 0 or inner_adj is None:
            inner_adj = update_adj(z)       
            GATE_loss, cluster_loss = pretrain_loss_function(data = pyg_data.x, recon = recon, z = z, adj = inner_adj, c = c_softmax, edge_index = pyg_data.edge_index, n_neighbors = NUM_LOSS_NEIGH)
            velo_loss, time_cor, temp_smooth_loss = post_loss_gene(gene_adj, device, device2, z, pyg_data, cp, up_type_features, t, c_softmax, n_neighbors=NUM_LOSS_NEIGH, warm_up = True, corr_mode = corr_mode, cos_batch = cos_batch)
            loss = GATE_loss + cluster_loss + velo_loss + time_cor + temp_smooth_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
        else:
            # Phase 5: Train the velocity
            recon, z, t, c_softmax, cp, up_type_features = model(pyg_data.x, pyg_data.edge_index, use_initial_regulation=False)
            #if epoch % update_interval == 0 or inner_adj is None:
            inner_adj = update_adj(z)     
            GATE_loss, cluster_loss = pretrain_loss_function(data = pyg_data.x, recon = recon, z = z, adj = inner_adj, c = c_softmax, edge_index = pyg_data.edge_index, n_neighbors = NUM_LOSS_NEIGH)
            velo_loss, time_cor, temp_smooth_loss = post_loss_gene(gene_adj, device, device2, z, pyg_data, cp, up_type_features, t, c_softmax, n_neighbors=NUM_LOSS_NEIGH, warm_up = False, corr_mode = corr_mode, cos_batch = cos_batch)
            loss = GATE_loss + cluster_loss + velo_loss + time_cor + temp_smooth_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            scheduler.step()
            current_loss = loss.item()
            if current_loss < best_loss - min_loss_improvement:
                best_loss = current_loss
                patience_counter = 0  # Reset patience counter on improvement
            else:
                patience_counter += 1  # Increment patience counter if no improvement
            
            # Check for early stopping condition
            if patience_counter > early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch} (patience counter: {patience_counter})")
                break  # Exit the training loop

        if epoch % log_interval == 0:
            print(f"Epoch {epoch}, Loss {loss.item()}")
            if epoch > pretrain_epochs + cluster_epochs + time_epochs:
                print(f"GATE: {GATE_loss}, Cluster: {cluster_loss}, Velo: {velo_loss}, Time_cor: {time_cor}, Time_smooth: {temp_smooth_loss}")
            elif epoch > pretrain_epochs + cluster_epochs:
                print(f"GATE: {GATE_loss}, Cluster: {cluster_loss}, Time_cor: {time_cor}, Time_smooth: {temp_smooth_loss}")
    
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        num_genes = adata.n_vars
        recon, z, t, c_softmax, cp, up_type_features = model(pyg_data.x, pyg_data.edge_index, use_initial_regulation=False)
        # Identify the cluster with the highest probability for each cell
        cluster = torch.argmax(c_softmax, dim=1)
        weight = torch.max(c_softmax, dim=1).values

        # Reshape cp to (num_cells, num_genes, 3)
        cp_reshaped = cp.view(recon.shape[0], num_genes, 3)
        alpha, beta, gamma = cp_reshaped[..., 0], cp_reshaped[..., 1], cp_reshaped[..., 2]

        # Convert tensors to numpy for further processing/storage
        orig = pyg_data.x.cpu().numpy()
        recon = recon.cpu().numpy()
        cluster = cluster.cpu().numpy()
        weight = weight.cpu().numpy()
        # cell_ids = cell_ids.cpu().numpy()
        # Assuming cell_ids can be either a tensor or a numpy array
        cell_ids = pyg_data.cell_ids.cpu().numpy() if isinstance(pyg_data.cell_ids, torch.Tensor) else pyg_data.cell_ids

        z_numpy = z.cpu().numpy()
        t = t.cpu().numpy()
        up_type_features = up_type_features.cpu().numpy()

        if MODEL_MODE == 'pretrain':
            # Add the results
            adata.layers['recon_u'] = recon[:,:adata.n_vars]
            adata.layers['recon_s'] = recon[:,adata.n_vars:]
            adata.layers['scale_Mu'] = orig[:,:adata.n_vars]
            adata.layers['scale_Ms'] = orig[:,adata.n_vars:]
            adata.layers['pred_cell_type'] = up_type_features
            adata.obs['pred_cluster'] = cluster
            adata.obs['pred_clus_weight'] = weight
            adata.obsm['X_pre_embed'] = z_numpy
            adata.obsm['cluster_matrix'] = c_softmax.cpu().numpy()
            # adata.uns['coar_adj'] = coar_adj.detach().cpu().numpy()
            # Calculate neighbors based on 'X_pre_embed' and store them in a new slot
            sc.pp.neighbors(adata,n_neighbors=30, use_rep='X_pre_embed', key_added='pre_embed_neighbors')
            # Calculate UMAP based on the new neighbors and store it in a new slot
            temp_adata = sc.tl.umap(adata, neighbors_key='pre_embed_neighbors', copy = True)
            adata.obsm['X_umap_pre_embed'] = temp_adata.obsm['X_umap']
            del temp_adata
            adata.obs['pred_cluster'] = adata.obs['pred_cluster'].astype('category')
        else:
            # Add the results
            adata.layers['recon_u_refine'] = recon[:,:adata.n_vars]
            adata.layers['recon_s_refine'] = recon[:,adata.n_vars:]
            adata.layers['scale_Mu_refine'] = orig[:,:adata.n_vars]
            adata.layers['scale_Ms_refine'] = orig[:,adata.n_vars:]
            adata.layers['pred_cell_type_refine'] = up_type_features
            adata.layers['recon_alpha'], adata.layers['recon_beta'], adata.layers['recon_gamma'] = alpha.cpu().numpy(), beta.cpu().numpy(), gamma.cpu().numpy()
            adata.layers['pred_vu'] = adata.layers['recon_alpha'] - adata.layers['recon_beta'] * adata.layers['scale_Mu_refine']
            adata.layers['pred_vs'] = adata.layers['recon_beta'] * adata.layers['scale_Mu_refine'] - adata.layers['recon_gamma'] * adata.layers['scale_Ms_refine']
            adata.obs['pred_cluster_refine'] = cluster
            adata.obs['pred_clus_weight_refine'] = weight
            adata.obs['pred_time'] = t
            adata.obsm['X_refine_embed'] = z_numpy
            adata.obsm['cluster_matrix'] = c_softmax.cpu().numpy()
            # adata.uns['coar_adj'] = coar_adj.detach().cpu().numpy()
            # Calculate neighbors based on 'X_refine_embed' and store them in a new slot
            sc.pp.neighbors(adata,n_neighbors=30, use_rep='X_refine_embed', key_added='refine_embed_neighbors')
            # Calculate UMAP based on the new neighbors and store it in a new slot
            temp_adata = sc.tl.umap(adata, neighbors_key='refine_embed_neighbors', copy = True)
            adata.obsm['X_umap_refine_embed'] = temp_adata.obsm['X_umap']
            del temp_adata
            adata.obs['pred_cluster_refine'] = adata.obs['pred_cluster_refine'].astype('category')

    # Return the DataFrame for the current gene
    return adata


# Function to train and evaluate for each gene
def pretrain_mclust(device, pyg_data, adata, NUM_LOSS_NEIGH, max_n_cluster, mask_train = False):

    # Set random seed inside the function
    SEED = 618
    torch.manual_seed(SEED)
    # Prepare data
    pyg_data = pyg_data.to(device)
    # Initialize the GATE model
    model = GATModel(
        in_features=pyg_data.x.shape[1],
        hid_features=512, #256, # raw 512
        out_features=128, # 2, # raw 128
        num_clusters=max_n_cluster,
        num_genes=adata.n_vars,
        type_features=pyg_data.type_features,
        use_experts=False
    ).to(device)
    pyg_data.unsplice = pyg_data.x[:,:adata.n_vars]
    pyg_data.splice = pyg_data.x[:,adata.n_vars:]

    # Define optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Training loop
    MAX_EPOCH = 3000

    log_interval = 50
    model.train()
    
    for epoch in range(MAX_EPOCH):
        optimizer.zero_grad()

        # Phase 1: Just train the GATE
        if mask_train:
            mask = generate_mask(pyg_data.x, 0.15)
            x_unmasked = pyg_data.x * (1 - mask)
            recon, z, t, c_softmax, cp, up_type_features = model(x_unmasked, pyg_data.edge_index, use_initial_regulation=True)
            GATE_loss, _ = pretrain_loss_function(data = pyg_data.x, recon = recon, z = z, adj = pyg_data.adj, c = c_softmax, edge_index = pyg_data.edge_index, n_neighbors = NUM_LOSS_NEIGH)
        else:
            recon, z, t, c_softmax, cp, up_type_features = model(pyg_data.x, pyg_data.edge_index, use_initial_regulation=True)
            GATE_loss, _ = pretrain_loss_function(data = pyg_data.x, recon = recon, z = z, adj = pyg_data.adj, c = c_softmax, edge_index = pyg_data.edge_index, n_neighbors = NUM_LOSS_NEIGH)
        loss = GATE_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()

        if epoch % log_interval == 0:
            print(f"Epoch {epoch}, Loss {loss.item()}")

    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        num_genes = adata.n_vars
        recon, z, t, c_softmax, cp, up_type_features = model(pyg_data.x, pyg_data.edge_index, use_initial_regulation=False)
        # Identify the cluster with the highest probability for each cell
        cluster = torch.argmax(c_softmax, dim=1)
        weight = torch.max(c_softmax, dim=1).values

        # Reshape cp to (num_cells, num_genes, 3)
        cp_reshaped = cp.view(recon.shape[0], num_genes, 3)
        alpha, beta, gamma = cp_reshaped[..., 0], cp_reshaped[..., 1], cp_reshaped[..., 2]

        # Convert tensors to numpy for further processing/storage
        orig = pyg_data.x.cpu().numpy()
        recon = recon.cpu().numpy()
        cluster = cluster.cpu().numpy()
        weight = weight.cpu().numpy()
        # cell_ids = cell_ids.cpu().numpy()
        # Assuming cell_ids can be either a tensor or a numpy array
        cell_ids = pyg_data.cell_ids.cpu().numpy() if isinstance(pyg_data.cell_ids, torch.Tensor) else pyg_data.cell_ids

        z_numpy = z.cpu().numpy()
        t = t.cpu().numpy()
        up_type_features = up_type_features.cpu().numpy()

        # Add the results
        adata.layers['recon_u'] = recon[:,:adata.n_vars]
        adata.layers['recon_s'] = recon[:,adata.n_vars:]
        adata.layers['scale_Mu'] = orig[:,:adata.n_vars]
        adata.layers['scale_Ms'] = orig[:,adata.n_vars:]
        adata.layers['pred_cell_type'] = up_type_features
        adata.obs['pred_cluster'] = cluster
        adata.obs['pred_clus_weight'] = weight
        adata.obsm['X_pre_embed'] = z_numpy
        adata.obsm['cluster_matrix'] = c_softmax.cpu().numpy()
        # adata.uns['coar_adj'] = coar_adj.detach().cpu().numpy()
        # Calculate neighbors based on 'X_pre_embed' and store them in a new slot
        sc.pp.neighbors(adata,n_neighbors=30, use_rep='X_pre_embed', key_added='pre_embed_neighbors')
        # Calculate UMAP based on the new neighbors and store it in a new slot
        temp_adata = sc.tl.umap(adata, neighbors_key='pre_embed_neighbors', copy = True)
        adata.obsm['X_umap_pre_embed'] = temp_adata.obsm['X_umap']
        del temp_adata
        adata.obs['pred_cluster'] = adata.obs['pred_cluster'].astype('category')

    # Return the DataFrame for the current gene
    return adata
