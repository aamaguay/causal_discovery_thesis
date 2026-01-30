import numpy as np
import torch
from torch import nn
from causa.utils import TensorDataLoader
from sklearn.preprocessing import StandardScaler, SplineTransformer
from causa.utils import load_flow_config

from causa.hsic import HSIC
from causa.het_ridge import convex_fgls
from causa.ml import map_optimization, mod_opt_joint_loglik, contruct_nn
from torch.nn.functional import relu, leaky_relu, elu
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


def compute_marginal_likelihood_nn(
    x,
    conf_name=None,
    flow_name=None,
    n_steps=None,
    custom_map_kwargs=None,
    seed=711,
    device="cpu",
):
    """
    Estimate the marginal log-likelihood of a one-dimensional variable x
    using a noise-augmented normalizing flow.

    The method augments x with independent Gaussian noise e ~ N(0, 1) to form
    a two-dimensional variable (x, e), trains a normalizing flow on the joint
    distribution p(x, e), and recovers the marginal density via:

        log p(x) = log p(x, e) - log p(e)

    The returned value is the average marginal log-likelihood per sample.

    Parameters
    ----------
    x : np.ndarray of shape (n,)
        One-dimensional input data.
    conf_name : str or None, default=None
        Name of a predefined flow configuration (e.g. "conf1").
    flow_name : str or None, default=None
        Name of the normalizing flow ("nsf", "realnvp").
    n_steps : int or None, default=None
        Number of training epochs for the flow (default: 5000).
    custom_map_kwargs : dict or None, default=None
        Custom configuration dictionary for the flow. Mutually exclusive
        with conf_name.
    seed : int, default=711
        Random seed for reproducibility.
    device : str, default="cpu"
        Device to use for training ("cpu" or "cuda").

    Returns
    -------
    marg_log_lik : float
        Estimated average marginal log-likelihood per sample for x.
    model : nn.Module
        Trained normalizing flow model on (x, e).
    """
    if flow_name is None:
        raise ValueError("flow_name must be provided for marginal likelihood estimation.")

    if conf_name is not None and custom_map_kwargs is not None:
        raise ValueError("Provide only one of conf_name or custom_map_kwargs.")

    n_steps = 5000 if n_steps is None else n_steps

    # Ensure correct shape: (n,)
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional array of shape (n,)")

    set_seed(seed)

    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    n = x_t.shape[0]

    # Augment with independent Gaussian noise
    noise = torch.randn(n, device=device)
    x_aug = torch.stack([x_t, noise], dim=1)  # shape (n, 2)

    # Load flow configuration
    map_kwargs = load_flow_config(
        flow_name=flow_name,
        conf_name=conf_name,
        custom_config=custom_map_kwargs,
        features=2,
    )

    # Data loader (full batch)
    loader = DataLoader(
        TensorDataset(x_aug),
        batch_size=n,
        shuffle=False,
    )

    # Construct and train flow
    model = contruct_nn(**map_kwargs)
    model, losses = mod_opt_joint_loglik(
        model=model,
        train_loader=loader,
        n_epochs=n_steps,
        lr=1e-2,
        lr_min=1e-6,
        optimizer="Adam",
        scheduler="exp",
    )

    log_p_xe = - np.nanmin(losses)

    # Exact log-density of Gaussian noise
    log_p_e = (-0.5 * noise.numpy()**2 - 0.5 * np.log(2 * np.pi)).sum()
    marg_log_lik = (log_p_xe - log_p_e) / n

    return marg_log_lik, model



def loci(
    x,
    y,
    independence_test=True,
    neural_network=True,
    return_function=False,
    n_steps_cond_prob=None,
    flow_name=None,
    conf_name=None,
    n_steps_marg_prob=None,
    marginal_loglik=False,
    custom_map_kwargs=None,
):
    """
    Location–Scale Causal Inference (LOCI) for bivariate causal discovery.

    LOCI evaluates the causal direction between two one-dimensional variables
    x and y by fitting heteroscedastic location–scale models in both directions
    (x → y and y → x) and comparing either:
      - residual independence (HSIC-based), or
      - likelihood-based scores.

    By convention:
        score > 0  → evidence for x → y  
        score < 0  → evidence for y → x

    Parameters
    ----------
    x : np.ndarray of shape (n,)
        First variable.
    y : np.ndarray of shape (n,)
        Second variable.
    independence_test : bool, default=True
        If True, use residual independence (HSIC) for scoring.
        If False, use likelihood-based scoring.
    neural_network : bool, default=True
        If True, use neural network-based heteroscedastic estimators.
        If False, use convex baseline estimators.
    return_function : bool, default=False
        If True, also return fitted conditional and marginal models.
    n_steps_cond_prob : int or None, default=None
        Training steps for conditional models.
    flow_name : str or None, default=None
        Name of the normalizing flow used for marginal likelihoods.
    conf_name : str or None, default=None
        Name of predefined flow configuration.
    n_steps_marg_prob : int or None, default=None
        Training steps for marginal likelihood estimation.
    marginal_loglik : bool, default=False
        If True, include marginal log-likelihood terms in the score.
        This yields an extension of original LOCI.
    custom_map_kwargs : dict or None, default=None
        Custom configuration dictionary for the flow.

    Returns
    -------
    score : float
        LOCI causal score.
    (optional)
    f_forward : callable
        Conditional mean/scale estimator for y | x.
    f_reverse : callable
        Conditional mean/scale estimator for x | y.
    f_forward_marg : nn.Module or None
        Marginal density model for x (if enabled).
    f_reverse_marg : nn.Module or None
        Marginal density model for y (if enabled).
    """

    assert x.ndim == y.ndim == 1, 'x and y have to be 1-dimensional arrays'

    # --------------------
    # Conditional models
    # --------------------
    if neural_network:
        log_lik_xy, f_xy = het_fit_nn(x, y, n_steps_cond_prob)
        log_lik_yx, f_yx = het_fit_nn(y, x, n_steps_cond_prob)
    else:
        log_lik_xy, f_xy = het_fit_convex(x, y, n_steps_cond_prob)
        log_lik_yx, f_yx = het_fit_convex(y, x, n_steps_cond_prob)

    # --------------------
    # Marginal likelihoods (optional)
    # --------------------
    log_marg_xy = 0.0
    log_marg_yx = 0.0
    f_marg_x = None
    f_marg_y = None

    if marginal_loglik:
        if flow_name is None:
            raise ValueError("flow_name must be provided when marginal_loglik=True")

        log_marg_xy, f_marg_x = compute_marginal_likelihood_nn(
            x,
            conf_name=conf_name,
            flow_name=flow_name,
            n_steps=n_steps_marg_prob,
            custom_map_kwargs=custom_map_kwargs,
        )

        log_marg_yx, f_marg_y = compute_marginal_likelihood_nn(
            y,
            conf_name=conf_name,
            flow_name=flow_name,
            n_steps=n_steps_marg_prob,
            custom_map_kwargs=custom_map_kwargs,
        )

    # --------------------
    # Scoring
    # --------------------
    if independence_test:
        my, sy = f_xy(x)
        mx, sx = f_yx(y)

        indep_xy = HSIC(x, (y - my) / sy)
        indep_yx = HSIC(y, (x - mx) / sx)

        score = indep_yx - indep_xy  # positive → x → y
    else:
        score = (log_lik_xy + log_marg_xy) - (log_lik_yx + log_marg_yx)

    # --------------------
    # Return
    # --------------------
    if return_function:
        return score, f_xy, f_yx, f_marg_x, f_marg_y

    return score
