import torch
import numpy as np
torch.set_default_tensor_type(torch.DoubleTensor)
def spl(phi, Ne, Nc_e,grid_interval):
    Nc=Ne+Nc_e
    # if phi.size(0)==Nc:
    #     print('yes1')
    epsilon=0
    mat=(torch.mm(phi,torch.transpose(phi,0,1))+torch.eye(Nc).cuda()*epsilon)*grid_interval**3 #add intervel
    # print(mat.shape,np.linalg.matrix_rank(mat.cpu().detach().numpy()))
    # if mat.size(0)==Nc:
    #     print('yes2')
    [SG1,SG2]=torch.split(mat,[Ne,Nc_e],0)  
    SG_11,SG_12= torch.split(SG1,[Ne,Nc_e],1) 
    SG_21,SG_22= torch.split(SG2,[Ne,Nc_e],1) 
    # print(SG_11.shape,Ne)
    # print(SG_12.shape,)
    # print(SG_21.shape)
    # print(SG_22.shape,Nc_e)
    return SG_11, SG_12, SG_21, SG_22
        
def get_orbitals(M, phi,grid_interval):
    Ne=M.size(1)
    Nc_e=M.size(0)
    SG_11, SG_12, SG_21, SG_22=spl(phi, Ne, Nc_e,grid_interval)    
    
    S=SG_11 + torch.mm(SG_12,M) + torch.mm(torch.transpose(M,0,1),SG_21) + torch.mm(torch.mm(torch.transpose(M,0,1),SG_22),M)
    # sq=torch.cholesky(S).inverse()
    [u,sg,v]=torch.svd(S)
    sq=(u@torch.diag(sg**(1/2))).inverse()
    # print(sq@S@sq.transpose(0,1))
    psi= sq@torch.cat((torch.eye(Ne).cuda(),M.transpose(0,1)),1)@phi
    # print(psi@psi.transpose(0,1))
    Ga1=SG_11+M.transpose(0,1)@SG_21
    Ga2=SG_12+M.transpose(0,1)@SG_22
    A= - Ga1.inverse() @ Ga2
    # print(A.shape,Ne, Nc_e)
    S_=SG_22 + torch.mm(SG_21,A) + torch.mm(torch.transpose(A,0,1),SG_12) + torch.mm(torch.mm(torch.transpose(A,0,1),SG_11),A)
    [u,sg,v]=torch.svd(S_)
    sq_=(u@torch.diag(sg**(1/2))).inverse()
    # print(sq_@S_@sq_.transpose(0,1))
    psi_cpl= sq_ @ torch.cat((torch.transpose(A,0,1),torch.eye(Nc_e).cuda()),1) @ phi
    return psi, psi_cpl

def get_orbitals_triple(M, phi, phi_l, phi_g, grid_interval):
    Ne=M.size(1)
    Nc_e=M.size(0)
    SG_11, SG_12, SG_21, SG_22=spl(phi, Ne, Nc_e, grid_interval)    
    
    S=SG_11 + torch.mm(SG_12,M) + torch.mm(torch.transpose(M,0,1),SG_21) + torch.mm(torch.mm(torch.transpose(M,0,1),SG_22),M)
    [u,sg,v]=torch.svd(S)
    sq=(u@torch.diag(sg**(1/2))).inverse()
    # print(sq@S@sq.transpose(0,1))
    Co=sq@torch.cat((torch.eye(Ne).cuda(),M.transpose(0,1)),1)
    psi= Co@phi
    psi_l= Co@phi_l
    psi_g= Co@phi_g
    # print(psi@psi.transpose(0,1))
    
    Ga1=SG_11+M.transpose(0,1)@SG_21
    Ga2=SG_12+M.transpose(0,1)@SG_22
    A= - Ga1.inverse() @ Ga2
    # print(A.shape,Ne, Nc_e)
    S_=SG_22 + torch.mm(SG_21,A) + torch.mm(torch.transpose(A,0,1),SG_12) + torch.mm(torch.mm(torch.transpose(A,0,1),SG_11),A)
    # epsilon=abs(S_).min()
    # S_eps=torch.eye(Nc_e).cuda()*epsilon+S_
    # print(S_.shape,np.linalg.matrix_rank(S_.cpu().detach().numpy()))
    [u,sg,v]=torch.svd(S_)
    sq_=(u@torch.diag(sg**(1/2))).inverse()
    Co_= sq_ @ torch.cat((torch.transpose(A,0,1),torch.eye(Nc_e).cuda()),1)
    psi_cpl= Co_ @ phi
    psi_cpl_l= Co_ @ phi_l
    return psi, psi_l, psi_g, psi_cpl ,psi_cpl_l


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU.')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU.')
    G=500
    Ne=46
    Nc=46+70 
    grid_interval=0.01
    M=torch.rand(Nc-Ne,Ne).cuda()
    phi=torch.rand(Nc,G).cuda()  
    psi, psi_cpl = get_orbitals(M,phi,grid_interval)
    print(psi@psi.transpose(0,1)*grid_interval**3)
    print(psi_cpl@psi_cpl.transpose(0,1)*grid_interval**3)
    print(psi@psi_cpl.transpose(0,1)*grid_interval**3)