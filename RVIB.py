import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import math

# A method to build a PyTorch dataset for training. Inputs X and the target Y is given.
# For this model, the recurrent neural network takes an input in 3dimension (n_samples, n_times, n_features)

class tDS(Dataset):
    def __init__(self,X,y,device='cpu',regression=True):
        if not isinstance(X,torch.Tensor):
            X = torch.Tensor(X)
        if not isinstance(y,torch.Tensor):
            y = torch.Tensor(y)
        self.X = X.to(dtype=torch.float,device=device)
        y_type = torch.long
        if regression:
            y_type = torch.float
        self.y = y.to(dtype=y_type,device=device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        xi = self.X[idx,:,:]
        yi = self.y[idx]
        return xi,yi


def get_dataloader(X,y,batch_size = 1,shuffle=False,device='cpu',regression=True):
    dataset = tDS(X,y,device=device,regression=regression)
    return DataLoader(dataset,batch_size = batch_size,shuffle=False)

# The classo of RVIB model which is based on a single layer of GRU and a final Dense layer for prediction.
class RVIB_GRU(nn.Module):
    def __init__(self, input_dim, time_dim, #Input dimension
                 z_dim, beta=1, #IB principle hyper-parmeters
                 fixed_r = False,mu_r=0.,logvar_r=0.):# Prior distribution assumptions p(Z) ~ N(mu_r,var_r)
        """

        Args:
            input_dim (int): Dimension for the features
            z_dim (int): Dimension for the latent distribution p(Z|X)
            time_dim (int): Time window size (W in paper)
            beta (float): Trade-off paramter in IB princple. None-negative value. The compression/penalization of the model is higher with large beta
            fixed_r (bool): The prior distribution parameters are also trained if set to True (see below mu and logvar).
            mu_r (float): Expectation for assumed prior distribution. Defaults to 0..
            logvar_r (float): Variance for assumed prior distributions. The variance matrix is also assumed to be diagonal Defaults to 0..
        """

        super(RVIB_GRU, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = z_dim*2 # The hidden variable is represented by the expectation and the log_var, thus need two times the dimension of Z.
        self.time_dim = time_dim
        self.beta = beta
        
        # self.rnn = nn.GRUCell(input_dim, self.hidden_dim)
        self.RNN = nn.GRU(input_size = input_dim,
                            hidden_size = self.hidden_dim,
                            num_layers=1,batch_first=True)
        
        # The linear layer mapping the hidden state to the target state. Suppose single dimensional outpu
        self.decoder = nn.Linear(z_dim, 1)
        # self.loss_f = nn.MSELoss() # Check if it is necessary
        
        # Check the convergence with below attributes
        self.loss_histo = []
        self.std_histo = []
        self.lh = nn.Parameter(None,requires_grad=False) # nn.Parameter allows to 'pickle' the model

        # Pivot distribution N(mu,sigma)
        self.register_parameter(name='mu_r',param = nn.Parameter(torch.tensor(mu_r)))
        self.register_parameter(name='logvar_r',param = nn.Parameter(torch.tensor(logvar_r)))
        
        if fixed_r:
            self.mu_r.requires_grad = False
            self.logvar_r.requires_grad = False
            
        # device indicatew wether the model is on CPU or GPU
        self.device='cpu'
    
    def cuda(self):
        self.device='cuda'
        return super().cuda()
    
    def cpd(self):
        self.device='cpu'
        return super().cpu()

    def rnn_forward(self,x,return_full=False):
        """

        Process the forward operation for the GRU parts

        Args:
            x (torch.Tensor) of dimesion (batch_size, time_dim, input_dim): The input sequence.
            return_full (bool): If True, the method returns the whole sequence of hidden states h.

        Returns:
            torch.Tensor: Last hidden state from the recurrent processing or the sequence of the hidden states regarding the time window.
        """
        h_full,h_last = self.RNN(input=x)
        h_last = h_last[0]
        res = h_last
        if return_full:
            res = h_full
        return res
        
    def encode(self,x,return_full=False):
        h = self.rnn_forward(x,return_full)
        mu,std = h[...,:self.z_dim],h[...,self.z_dim:]
        std = nn.functional.softplus(std,beta=20)
        return mu,std
    
    def decode(self,z):
        return self.decoder(z)
        
    def forward(self, x,return_full = False):
        mu,std = self.encode(x,return_full)
        mu_last = mu
        if return_full:
            mu_last = mu[:,-1,:]
        y_hat = self.decoder(mu_last).squeeze(dim=-1)
        return y_hat,mu,std
    
    def compute_kl(self,mu1,logvar1,mu2,logvar2):
        """
        Compute KL divergence KL(p_1 || p_2) between two gaussian multivariate distribution
        KL = 0.5*( \sum_{j=1}^{d}\log\frac{sigma_2^2 (j)}{sigma_1^2 (j)} - d 
                    + \sum_{j=1}^{d} \frac{(mu_1 (j)- mu_2 (j))^2 }{sigma_2^2 (j)}
                    + \sum_{j=1}^{d} \frac{sigma_1^2 (j)}{sigma_2^2 (j)}
                    )

        Args:
            mu1 (float): expectation of distribution p_1
            logvar1 (float): log of variance of distribution p_1
            mu2 (float): expectation of distribution p_2
            logvar2 (float): log of variance of distribution p_2

        Returns:
            float: KL value
        """
        
        dim = mu1.shape[-1]  
        
        L = logvar2 - logvar1
        D = torch.pow((mu1 - mu2),2) * torch.exp(-logvar2)
        T = (logvar1 - logvar2).exp()
        
        kl = 0.5*(-dim + torch.sum(L+D+T,dim=-1))
        
        return kl
    
    def compute_entropy_gaus(self,logvar):
        """
        Suppose the a multivariate gaussian distribution with diagonal cov matrix (given as a vector).
        Entropy is computed by: 
        \frac{d}{2} (1 + \log(2.pi.\sigma)) + \frac{1}{2} \log \sum_{i=1}^{d}\sigma_{i}^2 
        """
        
        dim = logvar.shape[-1]
        
        det_sigma = logvar.sum(dim=-1)
        c = dim/2 * (1 + math.log(2*math.pi))
        
        return 0.5*det_sigma + c 
    
    def encoding_loss(self,Mu,Std,reduction='mean'):
        
        # Preprocessing
        logvar = 2*torch.log(Std)
        
        # Dimension warning
        if Mu.ndim<3:
            print("Sequence Encoding warning : parameters seems not having time dimension")
        
        mu_shift = Mu.roll(-1,dims=1) # Take the expectation of the next time sequence
        logvar_shift = logvar.roll(-1,dims=1)

        # Compute the KL(p(Z|X_{1:w}) || p(Z|X_{1:w-1}))
        kl_t1 = self.compute_kl(mu_shift[:,:-1,:],logvar_shift[:,:-1,:],Mu[:,:-1,:],logvar[:,:-1,:])
        
        # Compute the KL(p(Z|X_{1}) || r(Z))
        kl_t0 = self.compute_kl(Mu[:,0,:],logvar[:,0,:],self.mu_r,self.logvar_r)
        
        kl_time = torch.cat([kl_t0.unsqueeze(dim=1),kl_t1],dim=1)

        # Reduction for batch size.
        if reduction == 'none':
            kl_time = kl_time
        elif reduction == 'mean':
            kl_time = kl_time.sum(dim=1).mean(dim=0)
        else:
            raise ValueError ("Reudction for KL loss is possible only for \'mean\' or \'none\' ")
        
        return kl_time
            
    def decoding_loss(self,y_hat,y,mu,std):
        """
        Closed form formula for the decoding loss
        A linear layer is assumed with the output distribution Y|Z ~ N(A.Z + \bias, cst).
        The expectation is used as the approximation
        E[Y] = E[E[Y|Z]] = E[A.Z+b] = A.mu + b = y_hat.
        
        decoding loss is then
        dec_loss = E[log(q(Y|Z))] 
            \propto -E[(A.Z+b)^2]
            \propto - [(y - A.mu)^2 + A^2. \sigma^2]
        """
        
        w = self.decoder.weight
        b = self.decoder.bias

        
        mse = torch.pow((y - y_hat),2)
        sigma_l = (w**2 * std**2).sum(dim=-1)[...,-1] # Sum over z_dim and taking the last time step

        l= (mse+sigma_l)
        
        return l.mean(dim=0)
    
    
    def evaluate(self,x,y,return_mse = False):
        """
        Computing all the losses.
        """
        
        y_hat,mu,std = self(x,return_full=True)
        m = self.decoding_loss(y_hat,y,mu,std)
        k = self.encoding_loss(mu,std)
        l = m+self.beta * k
        
        if return_mse:
            # Return MSE indicator instead of the decoding loss which is MSE + Variance-loss
            m = torch.pow((y - y_hat),2).mean(0)
        
        return l,m,k

    def warm_start(self,x_eval,y_eval,tol=1e3,lr=1,iter_max=20):
        """
        A technique to speed up the learning.
        Set the KL divergence low enough before starting the fit. proc.
        This is because we usually high value of KL if we initialize randomly
        """

        optim = torch.optim.Adam(self.parameters(),lr=lr)
        _,_,k = self.evaluate(x_eval,y_eval,return_mse=True)

        i = 1

        while k>tol:
            _,_,k = self.evaluate(x_eval,y_eval,return_mse=True)
            optim.zero_grad()
            k.backward()
            optim.step()
            
            i+=1
            if i>iter_max:
                print('Break')
                break

        print("KL divergence warm_start ended with KL loss=",k.item())
        
    
    def fit(self,train_loader,
            lr=1e-3,epoch=1,
            n_eval_samples = -1,warm_start = False):
        
        n_total_samples = len(train_loader)
        
        # Set evaluation data
        if n_eval_samples == -1:
            x_eval = train_loader.dataset.X
            y_eval = train_loader.dataset.y
        elif n_eval_samples>=0 and n_eval_samples<=n_total_samples:
            indices = torch.randperm(n_total_samples)[:n_eval_samples]
            x_eval = train_loader.dataset.X[indices]
            y_eval = train_loader.dataset.y[indices]


        if warm_start:
            self.warm_start(x_eval,y_eval)
        
        # Set optimizer
        optim = torch.optim.Adam(self.parameters(),lr=lr)

        pbar = tqdm(range(epoch))
        for e in pbar:
            for x,y in train_loader:
                l,_,_ = self.evaluate(x,y)
                
                optim.zero_grad()
                l.backward()
                optim.step()
                
                    # Measure performance on dataset
            with torch.no_grad():
                
                l,m,k = self.evaluate(x_eval,y_eval,return_mse=True)
                l,m,k = l.item(),m.item(),k.item()
                pbar.set_postfix({'Loss':l,'kl loss':k,'MSE': m,'beta':self.beta})
                self.loss_histo.append(l)
                _,_,std_eval = self(x_eval)
                self.std_histo.append(std_eval)

        # Saving loss historical to a pytorch parameter.
        self.lh = torch.nn.Parameter(torch.tensor(self.loss_histo),requires_grad=False)