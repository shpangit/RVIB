# RVIB

Recurrent Variational Information Bottleneck.
This model is a recurrent neural network trained with a loss function derived from the Information Bottleneck principle(IB).
\[\max I(Z,Y) - \beta I(Z,X_{1:W}) \]
For now, it is only one layer (one directional) GRU combined with a linear layer to obtain the target variable to adresse a regression problem.

PyTorch (tested on 1.8.0) is used for the Gated Recurrent Unit(GRU) architecture.
Inputs are:
- Input dimension (*input_dim*): The feature size for input data.
- Window size (*time_dim*): The size of the time-window.
- Latent variable dimension (*z_dim*): Dimension for the hidden latent $Z \mid X$ and $Z$
- Trade-off parameter in IB (*beta*): $\beta$ parameter in IB-loss.
- Parameters for prior distribution (*fixed_r,mu_r,logvar_r*): distribution characteristics for priori distribution $p(Z)$. If *fixed_r = False*, thes distribution paramters are also trained.

An example on a time series is in RVIB_example1.ipynb
