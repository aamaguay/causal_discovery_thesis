import numpy as np
import torch
from torch import nn
from causa.utils import TensorDataLoader
from sklearn.preprocessing import StandardScaler, SplineTransformer

from causa.hsic import HSIC
from causa.het_ridge import convex_fgls
from causa.ml import map_optimization, mod_opt_joint_loglik, contruct_nn
from torch.nn.functional import relu
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class HetSpindlyHead(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(1, 1, bias=False)
        self.lin2 = nn.Linear(1, 1, bias=False)
        self.lin2.weight.data.fill_(0.0)

    def forward(self, input):
        out1 = self.lin1(input[:, 0].unsqueeze(-1))
        out2 = torch.exp(self.lin2(input[:, 1].unsqueeze(-1)))
        return torch.cat([out1, out2], 1)


def build_het_network(in_dim=1):
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 2),
        HetSpindlyHead()
    )


def test_indep_fgls(Phi, Psi, x, y, w_1, w_2):
    eta_1 = Phi @ w_1
    eta_2 = - torch.abs(Psi) @ w_2
    scale = torch.sqrt(- 0.5 / eta_2)
    loc = - 0.5 * eta_1 / eta_2
    residuals = (y.flatten() - loc) / scale
    dhsic_res = HSIC(residuals.flatten().cpu().numpy(), x)
    return dhsic_res


def test_indep_nn(model, x, y):
    y = y.flatten()
    with torch.no_grad():
        f = model(x)
        eta_2 =  - f[:, 1] / 2
        scale = torch.sqrt(- 0.5 / eta_2)
        loc = - 0.5 * f[:, 0] / eta_2
        residuals = (y - loc) / scale
    return HSIC(residuals.cpu().flatten().numpy(), x.cpu().flatten().numpy())

    
def het_fit_nn(x, y, n_steps=None, seed=711, device='cpu'):
    """Fit heteroscedastic noise model with convex estimator using neural network.
    More precisely we fit y = f(x) + g(x) N with N Gaussian noise and return a joint
    function for f and g.

    Returns
    -------
    log_lik : float
        log likelihood of the fit
    f : method
        method that takes vector of x values and returns mean and standard deviation. 
    """
    n_steps = 5000 if n_steps is None else n_steps
    x, y = torch.from_numpy(x).double(), torch.from_numpy(y).double()
    map_kwargs = dict(
        scheduler='cos',
        lr=1e-2,
        lr_min=1e-6,
        n_epochs=n_steps,
        nu_noise_init=0.5,
        prior_prec=0.0  # makes it maximum likelihood
    )
    loader = TensorDataLoader(
        x.reshape(-1, 1).to(device), y.flatten().to(device), batch_size=len(x)
    )
    set_seed(seed)
    model, losses, _, _, _ = map_optimization(
        build_het_network().to(device).double(),
        loader,
        likelihood='heteroscedastic_regression',
        **map_kwargs
    )

    @torch.no_grad()
    def f(x_):
        x_ = torch.from_numpy(x_[:, np.newaxis]).double()
        f = model(x_)
        eta_2 =  - f[:, 1] / 2
        scale = torch.sqrt(- 0.5 / eta_2)
        loc = - 0.5 * f[:, 0] / eta_2
        return loc.squeeze().cpu().numpy(), scale.squeeze().cpu().numpy()
    log_lik = - np.nanmin(losses) / len(x)
    return log_lik, f


def het_fit_convex(x, y, n_steps=None):
    """Fit heteroscedastic noise model with convex estimator using splines x -> y.
    More precisely we fit y = f(x) + g(x) N with N Gaussian noise and return a joint
    function for f and g.

    Returns
    -------
    f : method
       method that takes vector of x values and returns mean and standard deviation. 
    """
    n_steps = 1000 if n_steps is None else n_steps
    feature_map = SplineTransformer(n_knots=25, degree=5)
    Phi_x = torch.from_numpy(feature_map.fit_transform(x[:, np.newaxis])).double()
    y = torch.from_numpy(y).double()
    w_1, w_2, _, nll = convex_fgls(Phi_x, Phi_x.abs(), y, delta_Phi=1e-5, delta_Psi=1e-5, n_steps=n_steps)

    @torch.no_grad()
    def f(x_):
        Phi_x_ = torch.from_numpy(feature_map.transform(x_[:, np.newaxis])).double()
        eta_1 = Phi_x_ @ w_1
        eta_2 = - torch.abs(Phi_x_) @ w_2
        scale = torch.sqrt(- 0.5 / eta_2)
        loc = - 0.5 * eta_1 / eta_2
        return loc.squeeze().cpu().numpy(), scale.squeeze().cpu().numpy()

    return -nll, f

    
def loci(x, y, independence_test=True, neural_network=True, return_function=False, n_steps=None):
    """Location Scale Causal Inference (LOCI) for bivariate pairs. By default,
    the method returns a score for the x -> y causal direction where above 0
    indicates evidence for it and negative values indicate y -> x.

    Note: data x, y should be standardized or preprocessed in some way.
    
    Parameters
    ----------
    x : np.ndarray
        cause/effect vector 1-dimensional
    y : np.ndarray
        cause/effect vector 1-dimensional
    independence_test : bool, optional
        whether to run subsequent independence test of residuals, by default True
    neural_network : bool, optional
        whether to use neural network heteroscedastic estimator, by default True
    return_function : bool, optional
        whether to return functions to predict mean/std in both directions, by default False
    n_steps : int, optional
        number of epochs to train neural network or steps to optimize convex model
    """
    assert x.ndim == y.ndim == 1, 'x and y have to be 1-dimensional arrays'
    if neural_network:
        log_lik_forward, f_forward = het_fit_nn(x, y, n_steps)
        log_lik_reverse, f_reverse = het_fit_nn(y, x, n_steps)
    else:
        log_lik_forward, f_forward = het_fit_convex(x, y, n_steps)
        log_lik_reverse, f_reverse = het_fit_convex(y, x, n_steps)

    if independence_test:
        my, sy = f_forward(x)
        indep_forward = HSIC(x, (y - my) / sy)
        mx, sx = f_reverse(y)
        indep_reverse = HSIC(y, (x - mx) / sx)
        score = indep_reverse - indep_forward
    else:
        score = log_lik_forward - log_lik_reverse

    if return_function:
        return score, f_forward, f_reverse
    return score


#############################################################################
def compute_marginal_likelihood_nn(x, n_steps=None, seed=711, device='cpu'):
    """
    Estimate the marginal log-likelihood of a 1D variable `x` using an 
    augmented normalizing flow (RealNVP-based), where noise is added 
    to project the 1D variable into a higher-dimensional space.

    The function:
    - Adds independent Gaussian noise to `x` to create a 2D input [x, e]
    - Trains a RealNVP normalizing flow model on the joint distribution p(x, e)
    - Uses the change-of-variable formula to estimate log p(x)
      via log p(x) = log p(x, e) - log p(e)
    - Returns the average marginal log-likelihood per sample

    Parameters
    ----------
    x : array-like of shape (n,)
        The 1D input data whose marginal density is to be estimated.
    n_steps : int, optional
        Number of training epochs (default is 5000).
    seed : int, default=711
        Random seed for reproducibility.
    device : str, default='cpu'
        Device to use for training ('cpu' or 'cuda').

    Returns
    -------
    marg_log_lik : float
        The estimated average marginal log-likelihood per sample for x.
    model : nn.Module
        The trained normalizing flow model (fitted on [x, noise]).
    new_x : torch.Tensor of shape (n, 2)
        The augmented data matrix used during training (original x and noise).
    """
    n_steps = 5000 if n_steps is None else n_steps

    # define new matrix with additional noise
    # generate random noise
    set_seed(seed)
    n = len(x) 
    noise = torch.randn(n)  # mean 0, std 1 by default
    new_x = torch.tensor(np.stack([x, noise], axis=1), dtype=torch.float32)
    
    # define parameters for the flow design
    map_kwargs = dict(
        features                = new_x.shape[1],
        hidden_features         = 64,
        num_layers              = 6,
        num_blocks_per_layer    = 4,
        use_volume_preserving   = False, # False: affinity coupling
        activation              = relu,
        dropout_probability     = 0,
        batch_norm_within_layers= False,
        batch_norm_between_layers=False
    )
    
    # define data batches
    loader = DataLoader(
        TensorDataset(new_x), 
        batch_size=len(x),
        shuffle=False
    )
        
    # set flow and nn parameter
    flow_nn = contruct_nn(**map_kwargs)

    # estimation
    model, losses = mod_opt_joint_loglik(
                        model           = flow_nn,
                        train_loader    = loader,
                        n_epochs        = n_steps,
                        lr              = 1e-2,
                        lr_min          = 1e-6,
                        optimizer       = 'Adam',
                        scheduler       = 'exp'
    )

    joint_log_lik = - np.nanmin(losses)
    log_prob_noise = (-0.5 * noise.numpy()**2 - 0.5 * np.log(2 * np.pi)).sum()
    marg_log_lik = (joint_log_lik - log_prob_noise) / len(x)

    return marg_log_lik, model, new_x 

def loci_w_marginal(x, y, independence_test=True, 
                    neural_network=True, return_function=False, 
                    n_steps=None,
                    marginal_loglik = False):
    """Location Scale Causal Inference (LOCI) for bivariate pairs. By default,
    the method returns a score for the x -> y causal direction where above 0
    indicates evidence for it and negative values indicate y -> x.

    Note: data x, y should be standardized or preprocessed in some way.
    
    Parameters
    ----------
    x : np.ndarray
        cause/effect vector 1-dimensional
    y : np.ndarray
        cause/effect vector 1-dimensional
    independence_test : bool, optional
        whether to run subsequent independence test of residuals, by default True
    neural_network : bool, optional
        whether to use neural network heteroscedastic estimator, by default True
    return_function : bool, optional
        whether to return functions to predict mean/std in both directions, by default False
    n_steps : int, optional
        number of epochs to train neural network or steps to optimize convex model
    """
    assert x.ndim == y.ndim == 1, 'x and y have to be 1-dimensional arrays'
    if neural_network:
        log_lik_forward, f_forward = het_fit_nn(x, y, n_steps)
        log_lik_reverse, f_reverse = het_fit_nn(y, x, n_steps)
    else:
        log_lik_forward, f_forward = het_fit_convex(x, y, n_steps)
        log_lik_reverse, f_reverse = het_fit_convex(y, x, n_steps)
    
    if marginal_loglik:
        log_marg_lik_forward, model_f, new_x_f = compute_marginal_likelihood_nn(x = x, n_steps = n_steps)
        log_marg_lik_reverse, model_r, new_x_r = compute_marginal_likelihood_nn(x = y, n_steps = n_steps)
    else:
        log_marg_lik_forward = 0
        log_marg_lik_reverse = 0


    if independence_test:
        my, sy = f_forward(x)
        indep_forward = HSIC(x, (y - my) / sy)
        mx, sx = f_reverse(y)
        indep_reverse = HSIC(y, (x - mx) / sx)
        score = indep_reverse - indep_forward
    else:
        score_orig = (log_lik_forward ) - (log_lik_reverse )
        score_new = (log_lik_forward + log_marg_lik_forward) - (log_lik_reverse + log_marg_lik_reverse)

    if return_function:
        return score_new, score_orig, model_f, model_r, new_x_f, new_x_r, f_forward, f_reverse
    return score_new, score_orig
