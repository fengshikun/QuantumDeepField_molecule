import torch
import torch.nn as nn
import copy
import math
import torch.nn.functional as F
"""
directly from https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
"""

def gram_schmidt(vv):
    def projection(u, v):
#        return (v * u).sum() / (u * u).sum() * u
        return (v * u).sum() * u #(u * u).sum()=1

    nk = vv.size(0)
    uu = torch.zeros_like(vv, device=vv.device)
#    uu[0, :] = vv[0, :].clone()
    uu[0, :] = vv[0, :].clone() / vv[0, :].norm()
    for k in range(1, nk):
        vk = vv[k].clone()
        umat = uu[0:k, :].clone()
        mv = torch.mv(umat, vk)
        uk = vk - torch.mv(torch.transpose(umat,0,1), mv)
        uu[k, :] = uk / uk.norm()
#        uk = 0
#        for j in range(0, k):
#            uj = uu[j, :].clone()
#            uk = uk + projection(uj, vk)
#        uu[k, :] = vk - uk
#    for k in range(nk):
#        uk = uu[k, :].clone()
#        uu[k, :] = uk / uk.norm()
    return uu

def gram_schmidt_triple(vv,vv1, vv2, grid_interval):
    
    nk = vv.size(0) #Ne
    uu = torch.zeros_like(vv, device=vv.device) #Ne X G
    uu1 = torch.zeros_like(vv1, device=vv.device)
    uu2 = torch.zeros_like(vv2, device=vv.device)
    denom = vv[0, :].norm() * grid_interval**(3/2)
    uu[0, :] = vv[0, :].clone() / denom
    uu1[0, :] = vv1[0, :].clone()/ denom
    uu2[0, :] = vv2[0, :].clone()/ denom
    for k in range(1, nk):
        vk = vv[k].clone()
        vk1 = vv1[k].clone()
        vk2 = vv2[k].clone()
        umat = uu[0:k, :].clone()
        umat1 = uu1[0:k, :].clone()
        umat2 = uu2[0:k, :].clone()
        mv = torch.mv(umat, vk) * grid_interval**3
        uk = vk - torch.mv(torch.transpose(umat,0,1), mv)
        uk1 = vk1 - torch.mv(torch.transpose(umat1,0,1), mv)
        uk2 = vk2 - torch.mv(torch.transpose(umat2,0,1), mv)
        denom = uk.norm() * grid_interval**(3/2)
        uu[k, :] = uk / denom
        uu1[k, :] = uk1 / denom
        uu2[k, :] = uk2 / denom      
        
#    print(torch.mm(uu,torch.transpose(uu,0,1)))  # othognal check.
    return uu, uu1, uu2

def gram_schmidt_pair(vv,vv1, vv2):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    nk = vv.size(0)
    uu = torch.zeros_like(vv, device=vv.device)
    uu1 = torch.zeros_like(vv1, device=vv.device)
    uu2 = torch.zeros_like(vv2, device=vv.device)
    uu[0, :] = vv[0, :].clone()
    uu1[0, :] = vv1[0, :].clone()
    uu2[0, :] = vv2[0, :].clone()
    for k in range(1, nk):
        vk = vv[k].clone()
        vk1 = vv1[k].clone()
        vk2 = vv2[k].clone()
        uk = 0
        uk1 = 0
        uk2 = 0
        for j in range(0, k):
            uj = uu[j, :].clone()
            uk = uk + projection(uj, vk)
            uk1 = uk1 + projection(uj, vk1) 
            uk2 = uk2 + projection(uj, vk2) # diff
        uu[k, :] = vk - uk
        uu1[k, :] = vk1 - uk1
        uu2[k, :] = vk2 - uk2

    for k in range(nk):
        uk = uu[k, :].clone()
        uu[k, :] = uk / uk.norm()
        
        uk1 = uu1[k, :].clone()
        uu1[k, :] = uk1 / uk.norm()

        uk2 = uu2[k, :].clone()
        uu2[k, :] = uk2 / uk.norm()#顺序?
        
#    print(torch.mm(uu,torch.transpose(uu,0,1)))  # othognal check.
    return uu, uu1, uu2


"""
Get the orbital from phi lap_phi and coeff
phi: Nc X G
lap_phi: Nc X G
coeff: Ne x Nc

return:
psi: Ne X G
psi_lap: Ne X G
psi: 
"""
def get_orbital(coeff, phi_gto, phi_gto_gra, phi_gto_lap, grid_interval):
    psi_zero = torch.matmul(coeff, phi_gto)
    psi_zero_lap = torch.matmul(coeff, phi_gto_lap)
    psi_zero_gra = torch.matmul(coeff, phi_gto_gra)
    psi, psi_gra, psi_lap = gram_schmidt_triple(psi_zero, psi_zero_gra, psi_zero_lap, grid_interval)
    return psi, psi_gra, psi_lap


    # Ne = psi_zero_lap.shape[0]

    
    # psi = torch.zeros_like(psi_zero)
    # psi_lap = torch.zeros_like(psi_zero_lap)
    # psi[0] = psi_zero[0] / torch.norm(psi_zero[0], p=2)
    # psi_lap[0] = psi_zero_lap[0] / torch.norm(psi_zero_lap[0], p=2)
    # v = torch.matmul(psi[:1], psi_zero[0])
    # for i in range(1, Ne):
    #     # if i == 10:
    #     #     print("debug")
    #     psi_i = psi_zero[i] - torch.matmul(v, psi[:i])
    #     psi_lap_i = psi_zero_lap[i] - torch.matmul(v, psi_lap[:i])

    #     psi[i] = psi_i / torch.norm(psi_i, p=2)
    #     psi_lap[i] = psi_lap_i / torch.norm(psi_lap_i, p=2)

    #     v = torch.matmul(psi_zero[i], torch.transpose(psi[:i+1], 0, 1))
    
    # return psi, psi_lap

    # psi = gram_schmidt(psi_zero)
    # psi_lap = gram_schmidt(psi_zero_lap)
    # return psi, psi_lap

"""
Get the complemental orbital(lap)
"""
def get_complement_orbital(phi_gto, phi_gto_lap, psi, psi_lap, grid_interval):
    mat_phi_psi = torch.matmul(phi_gto, torch.transpose(psi, 0, 1)) * grid_interval**3
    psi_cpl = phi_gto - torch.matmul(mat_phi_psi, psi) # Nc X G
    psi_lap_cpl = phi_gto_lap - torch.matmul(mat_phi_psi, psi_lap) # Nc X G
    # print("check cpl zero", torch.matmul(psi_cpl, torch.transpose(psi, 0, 1)) * grid_interval**3)

    # keep non_zero (Nc - Ne) rows of psi_cpl and the corresponding rows of psi_lap_cpl
    # keep top-k norm rows
    psi_cpl_norm = torch.norm(psi_cpl, dim=1) * grid_interval**(3/2)
    psi_lap_cpl_norm = torch.norm(psi_lap_cpl, dim=1) * grid_interval**(3/2)
    
    #normalization
    k = phi_gto.shape[0] - psi.shape[0] # Nc - Ne
    assert k > 0
    topk_norms, topk_indices = torch.topk(psi_cpl_norm, k)
    psi_cpl_n = psi_cpl[topk_indices] / psi_cpl_norm[topk_indices].reshape([-1, 1]).repeat(1, psi_cpl.shape[1])
    psi_lap_cpl_n = psi_lap_cpl[topk_indices] /psi_lap_cpl_norm[topk_indices].reshape([-1, 1]).repeat(1, psi_lap_cpl.shape[1])
    
    # psi_cpl_n = psi_cpl[topk_indices] / psi_cpl_norm.expand_as(k,psi_cpl.shape[1])
    # psi_lap_cpl_n = psi_lap_cpl[topk_indices] / psi_lap_cpl_norm.expand_as(k,psi_lap_cpl.shape[1])
    # print('psi_cpl_n ,(Nc - Ne) *G',psi_cpl_n.shape,psi_lap_cpl_n.shape)
    # print("check cpl2 identity", torch.matmul(psi_cpl_n, torch.transpose(psi, 0, 1)) * grid_interval**3)
    return psi_cpl_n, psi_lap_cpl_n

"""
span psi(psi_cpl), V_xch to V(V_cpl)
V_xch: G
psi: Ne X G
return
V: Ne X Ne
"""
def getV(psi, V_xch):
    Ne = psi.shape[0]
    V_extend = V_xch.repeat(1, Ne)
    return torch.matmul(V_extend.transpose(0, 1) * psi, torch.transpose(psi, 0, 1))


"""
Diagonalization constraint on V + K
V: Ne X Ne
K: Ne X Ne
"""
def diag_constraint_loss(V, K):
    VK = V + K
    VK_abs = torch.abs(VK)
    VK_abs_diag = torch.diagonal(VK_abs)
    all_sum = VK.shape[0] *(VK.shape[0] - 1)
    return (torch.sum(VK_abs) - torch.sum(VK_abs_diag)) / all_sum

"""
Ground state constraint between V + K and V_cpl + K_cpl
V: Ne X Ne
K: Ne X Ne
V_cpl: Ne X Ne
K_cpl: Ne X Ne
"""

def get_constraint_loss(V, K, V_cpl, K_cpl):
    VK = V + K
    VK_cpl = V_cpl + K_cpl
    vk_diag = torch.diagonal(VK)
    vk_cpl_diag = torch.diagonal(VK_cpl)
    diff = torch.max(vk_diag) - torch.min(vk_cpl_diag)
    zero = torch.zeros_like(diff)
    return torch.maximum(diff, zero)



# padding matrix:
def padding_matrix(mat, max_dim, padding_value=0):
    cur_dim = mat.shape[1]
    if max_dim > cur_dim:
        new_mat = torch.randn([mat.shape[0], max_dim])
        new_mat[:, :cur_dim] = mat[:,:]
        new_mat[:, cur_dim:max_dim] = padding_value
        return new_mat.cuda()
    else:
        return mat[:, :max_dim]

def split_matrix(mat, row_array, col_array):
    mat_list = []
    i, j = 0, 0
    for idx, m in enumerate(row_array):
        n = col_array[idx]
        mat_list.append(mat[i:i+m, j:j+n])
        i += m
        j += n
    return mat_list




class AttentNelectron(nn.Module):
    def __init__(self, G, Nc, layer_num=1):
        super(AttentNelectron, self).__init__()
        self.G = G
        self.layer_num = layer_num
        self.Q_pre_func = nn.Linear(G, Nc)
        self.Q_functional = nn.ModuleList([nn.Linear(Nc, Nc)
                                           for _ in range(layer_num)])
        self.K_pre_func = nn.Linear(G, Nc)
        self.K_functional = nn.ModuleList([nn.Linear(Nc, Nc)
                                           for _ in range(layer_num)])

        self.V_pre_func = nn.Linear(G, Nc)
        self.V_functional = nn.ModuleList([nn.Linear(Nc, Nc)
                                           for _ in range(layer_num)])
        

    # phi: Ncut X G
    # Padding G to the same
    # Caution: because Nc is also sample-wise different, so it's hard to do batch propogation
    def forward(self, phi, Ne):
        phi = padding_matrix(phi, self.G)
        Q = self.Q_pre_func(phi)

        # Q = torch.relu(self.Q_pre_func(phi))
        # for l in range(self.layer_num):
        #     Q = torch.relu(self.Q_functional[l](Q))
        
        K = self.K_pre_func(phi)
        # K = torch.relu(self.K_pre_func(phi))
        # for l in range(self.layer_num):
        #     K = torch.relu(self.K_functional[l](Q))
        
        V = self.V_pre_func(phi)
        # V = torch.relu(self.V_pre_func(phi))
        # for l in range(self.layer_num):
        #     V = torch.relu(self.V_functional[l](Q))

        N_cut = V.shape[0]
        temperature = N_cut ** 0.5
        attn = torch.matmul(Q / temperature , K.transpose(0, 1))

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, V)
        output_clip = output[:, :int(Ne)]

        # values, indices = torch.topk(attn, int(Ne))
        
        
        # coeff = torch.randn([N_cut, int(Ne)]).cuda()
        # for idx in range(N_cut):
        #     coeff[idx] = V[idx][indices[idx]]
        return output_clip.transpose(0, 1) # Ne x N_cut

        

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product SelfAttention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        """
        :param query:
        :param key:
        :param value:
        :param mask:
        :param dropout:
        :return:
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    The multi-head attention module. Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1, bias=False):
        """

        :param h:
        :param d_model:
        :param dropout:
        :param bias:
        """
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h  # number of heads

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])  # why 3: query, key, value
        self.output_linear = nn.Linear(d_model, d_model, bias)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """

        :param query:
        :param key:
        :param value:
        :param mask:
        :return:
        """
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, _ = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class PositionwiseFeedForward(nn.Module): 
	def __init__(self, d_model, d_ff, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)
	
	def forward(self, x):
		return self.w_2(self.dropout(F.relu(self.w_1(x))))

def clones(module, N):
	# 克隆N个完全相同的SubLayer，使用了copy.deepcopy
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
	"Encoder是N个EncoderLayer的stack"
	def __init__(self, layer, N):
		super(Encoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)
	
	def forward(self, x, mask):
		for layer in self.layers:
			x = layer(x, mask)
		return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
	"""
	LayerNorm + sublayer(Self-Attenion/Dense) + dropout + 残差连接
	"""
	def __init__(self, size, dropout):
		super(SublayerConnection, self).__init__()
		self.norm = LayerNorm(size)
		self.dropout = nn.Dropout(dropout)
	
	def forward(self, x, sublayer):
		return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
	"EncoderLayer由self-attn和feed forward组成"
	def __init__(self, size, self_attn, feed_forward, dropout):
		super(EncoderLayer, self).__init__()
		self.self_attn = self_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 2)
		self.size = size

	def forward(self, x, mask):
		"Follow Figure 1 (left) for connections."
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
		return self.sublayer[1](x, self.feed_forward)

class PositionalEncoding(nn.Module): 
	def __init__(self, d_model, dropout, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		
		# Compute the positional encodings once in log space.
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) *
			-(math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:, :x.size(1)].detach()
		return self.dropout(x)


class AttentNelectronMHA(nn.Module):
    """
    G and Ne are padding number
    """
    def __init__(self, G, Ne, hidden_dim=512, d_ff=2048, head_num=8, mha_num=2, dropout=0.1):
        super(AttentNelectronMHA, self).__init__()
        self.G = G
        self.Ne = Ne
        self.pre_linear = nn.Linear(G, hidden_dim)
        self.mha_num = mha_num

        c = copy.deepcopy
        attn = MultiHeadedAttention(head_num, hidden_dim)
        ff = PositionwiseFeedForward(hidden_dim, d_ff, dropout)
        
        self.position = PositionalEncoding(hidden_dim, dropout)
        self.encoder = Encoder(EncoderLayer(hidden_dim, c(attn), c(ff), dropout), mha_num)
        # self.Q_functional = nn.ModuleList([MultiHeadedAttention(head_num, hidden_dim)
        #                                    for _ in range(mha_num)])
        self.post_linear = nn.Linear(hidden_dim, Ne)

        
    
    def reset_params(self):
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
        

    # phi shape: Bs x N_cut x G
    def forward(self, phi):
        x = self.pre_linear(phi)
        x = self.position(x)
        # mask is none
        x = self.encoder(x, None)
        x = self.post_linear(x)
        return x





if __name__ == "__main__":
    a = torch.randn(5, 6, requires_grad=True)
    c = torch.randn(5, 6, requires_grad=True)
    b = gram_schmidt(a)
    gram_schmidt_pair(a, c)
    coeff, phi_gto, phi_gto_lap = torch.randn((3, 4)), torch.randn((4, 4)), torch.randn((4, 4))
    get_orbital(coeff, phi_gto, phi_gto_lap)
