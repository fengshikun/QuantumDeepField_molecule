import torch
import torch.nn as nn
import torch.nn.functional as F
"""
directly from https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
"""

def gram_schmidt(vv):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    nk = vv.size(0)
    uu = torch.zeros_like(vv, device=vv.device)
    uu[0, :] = vv[0, :].clone()
    for k in range(1, nk):
        vk = vv[k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[j, :].clone()
            uk = uk + projection(uj, vk)
        uu[k, :] = vk - uk
    for k in range(nk):
        uk = uu[k, :].clone()
        uu[k, :] = uk / uk.norm()
    return uu


def gram_schmidt_pair(vv, vv2):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    nk = vv.size(0)
    uu = torch.zeros_like(vv, device=vv.device)
    uu2 = torch.zeros_like(vv2, device=vv.device)
    uu[0, :] = vv[0, :].clone()
    uu2[0, :] = vv2[0, :].clone()
    for k in range(1, nk):
        vk = vv[k].clone()
        vk2 = vv2[k].clone()
        uk = 0
        uk2 = 0
        for j in range(0, k):
            uj = uu[j, :].clone()
            uk = uk + projection(uj, vk)
            uk2 = uk2 + projection(uj, vk2) # diff
        uu[k, :] = vk - uk
        uu2[k, :] = vk2 - uk2

    for k in range(nk):
        uk = uu[k, :].clone()
        uu[k, :] = uk / uk.norm()

        uk2 = uu2[k, :].clone()
        uu2[k, :] = uk2 / uk2.norm()

    return uu, uu2


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
def get_orbital(coeff, phi_gto, phi_gto_lap):
    psi_zero = torch.matmul(coeff, phi_gto)
    psi_zero_lap = torch.matmul(coeff, phi_gto_lap)
    psi, psi_lap = gram_schmidt_pair(psi_zero, psi_zero_lap)
    return psi, psi_lap


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
def get_complement_orbital(phi_gto, phi_gto_lap, psi, psi_lap):
    mat_phi_psi = torch.matmul(phi_gto, torch.transpose(psi, 0, 1))
    psi_cpl = phi_gto - torch.matmul(mat_phi_psi, psi) # Nc X G
    psi_lap_cpl = phi_gto_lap - torch.matmul(mat_phi_psi, psi_lap) # Nc X G


    # keep non_zero (Nc - Ne) rows of psi_cpl and the corresponding rows of psi_lap_cpl
    # keep top-k norm rows
    psi_cpl_norm = torch.norm(psi_cpl, dim=1)
    
    k = phi_gto.shape[0] - psi.shape[0] # Nc - Ne
    assert k > 0
    topk_norms, topk_indices = torch.topk(psi_cpl_norm, k)
    return psi_cpl[topk_indices], psi_lap_cpl[topk_indices]

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

        

if __name__ == "__main__":
    a = torch.randn(5, 6, requires_grad=True)
    c = torch.randn(5, 6, requires_grad=True)
    b = gram_schmidt(a)
    gram_schmidt_pair(a, c)
    coeff, phi_gto, phi_gto_lap = torch.randn((3, 4)), torch.randn((4, 4)), torch.randn((4, 4))
    get_orbital(coeff, phi_gto, phi_gto_lap)
