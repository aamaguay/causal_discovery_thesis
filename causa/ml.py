import numpy as np
import torch
from torch.nn.utils import parameters_to_vector
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.distributions import Normal
import logging
from nflows.flows import SimpleRealNVP
from torch.nn.functional import relu


def heteroscedastic_normal(f, y, nu_noise=0.0, param_1='natural', param_2='additive'):
    # f should be two-dimensional per data point
    assert f.shape[0] == y.shape[0]
    assert y.ndim == 1
    assert f.shape[1] == 2
    assert nu_noise <= 0.0

    if param_2 == 'additive':
        f2 = f[:, 1] + nu_noise
        scale = torch.sqrt(- 0.5 / f2)
    elif param_2 == 'multiplicative':
        assert all(f[:, 1] >= 0)
        f2 = f[:, 1] * nu_noise
        scale = torch.sqrt(- 0.5 / f2)
    elif param_2 == 'stdev':
        scale = f[:, 1]

    if param_1 == 'natural':
        loc =  - 0.5 * f[:, 0] / f2
    elif param_1 == 'mean':
        loc = f[:, 0]

    dist = Normal(loc=loc, scale=scale)
    return -dist.log_prob(y).sum()


def map_optimization(model,
                     train_loader,
                     likelihood='regression',
                     param_1='natural',
                     param_2='multiplicative',
                     prior_prec=1.,
                     n_epochs=500,
                     lr=1e-3,
                     lr_min=None,
                     nu_noise_init=0.5,
                     optimizer='Adam',
                     scheduler='exp'):
    if lr_min is None:  # don't decay lr
        lr_min = lr
    device = parameters_to_vector(model.parameters()).device
    N = len(train_loader.dataset)

    # prior precision
    prior_prec = prior_prec * torch.ones(1, device=device)

    # set up loss (and observation noise hyperparam)
    if likelihood == 'classification':
        criterion = CrossEntropyLoss(reduction='mean')
    elif likelihood == 'regression':
        def criterion(f, y):
            lh = Normal(f, scale=1.0)
            return -lh.log_prob(y).mean()
    elif likelihood == 'heteroscedastic_regression':
        criterion = heteroscedastic_normal
        nu_noise = - torch.tensor(nu_noise_init, device=device)

    # set up model optimizer
    if optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f'Invalid optimizer {optimizer}')

    # set up scheduler for lr decay
    n_steps = n_epochs * len(train_loader)
    if scheduler == 'exp':
        min_lr_factor = lr_min / lr
        gamma = np.exp(np.log(min_lr_factor) / n_steps)
        scheduler = ExponentialLR(optimizer, gamma=gamma)
    elif scheduler == 'cos':
        scheduler = CosineAnnealingLR(optimizer, n_steps, eta_min=lr_min)
    else:
        raise ValueError(f'Invalid scheduler {scheduler}')

    losses = list()
    valid_perfs = list()
    valid_nlls = list()
    f2s = list()

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        epoch_perf = 0

        # standard NN training per batch
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            theta = parameters_to_vector(model.parameters())
            f = model(X)
            if prior_prec == 0.0:
                neg_log_prior = 0
            else:
                prior = Normal(torch.zeros_like(theta), scale=torch.ones_like(theta) / (prior_prec.sqrt()))
                neg_log_prior = - prior.log_prob(theta).sum()
            
            if likelihood == 'heteroscedastic_regression':
                loss = (criterion(f, y, nu_noise, param_1=param_1, param_2=param_2)
                        + neg_log_prior) / N
            else:
                loss = criterion(f, y) + neg_log_prior / N
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu().item() / len(train_loader)
            if likelihood == 'classification':
                epoch_perf += torch.sum(torch.argmax(f.detach(), dim=-1) == y).item() / N
            else:
                if likelihood == 'heteroscedastic_regression':
                    if param_1 == 'natural':
                        if param_2 == 'additive':
                            mean =  - 0.5 * f[:, 0] / (f[:, 1] + nu_noise)
                        elif param_2 == 'multiplicative':
                            mean = - 0.5 * f[:, 0] / (f[:, 1] * nu_noise)
                    elif param_1 == 'mean':
                        mean = f[:, 0]
                    else:
                        raise ValueError('Invalid parameterization param_1')
                else:
                    mean = f
                epoch_perf += (mean.detach() - y).square().sum() / N
            scheduler.step()
        losses.append(epoch_loss * N)
        if likelihood == 'heteroscedastic_regression':
            f2s.append(f[:, 1].mean().item())
        else:
            f2s.append(0)


        # compute validation error to report during training
        logging.info(f'MAP[epoch={epoch}]: network training. Loss={losses[-1]:.3f}; '
                        + f'Perf={epoch_perf:.3f}; lr={scheduler.get_last_lr()[0]:.7f}')

    return model, losses, valid_perfs, valid_nlls, f2s



########################################################################################
def contruct_nn(flow_name, features, hidden_features, num_layers, num_blocks_per_layer,
                use_volume_preserving = False, batch_norm_within_layers = False,
                num_bins = 10, tails = "linear", tail_bound = 3,
                apply_unconditional_transform = False,
                batch_norm_between_layers = False,
                activation = relu, dropout_probability = 0):
    
    try:
        if flow_name.lower() == 'realnvp':
            model_flow_ = SimpleRealNVP(
                features                = features,
                hidden_features         = hidden_features,
                num_layers              = num_layers,
                num_blocks_per_layer    = num_blocks_per_layer,
                use_volume_preserving   = use_volume_preserving, # False: affinity coupling
                activation              = activation,
                dropout_probability     = dropout_probability,
                batch_norm_within_layers= batch_norm_within_layers,
                batch_norm_between_layers=batch_norm_between_layers
            )
        elif flow_name.lower() == 'nsf':
            model_flow_ = SimpleNSF(
                features                = features,
                hidden_features         = hidden_features,
                num_layers              = num_layers,
                num_blocks_per_layer    = num_blocks_per_layer,
                num_bins                = num_bins,
                tails                   = tails,
                tail_bound              = tail_bound,
                activation              = activation,
                dropout_probability     = dropout_probability,
                apply_unconditional_transform = apply_unconditional_transform,
                batch_norm_within_layers= batch_norm_within_layers,
                batch_norm_between_layers=batch_norm_between_layers
            )
    except Exception as e:
        print(f"this {flow_name} flow is not supported..")
        print(e)

    return model_flow_

def mod_opt_joint_loglik(model,
            train_loader,
            n_epochs=500,
            lr=1e-3,
            lr_min=None,
            optimizer='Adam',
            scheduler='exp'):
    
    # assert lr value
    if lr_min is None:  # don't decay lr
        lr_min = lr

    # number of obs.
    N = len(train_loader.dataset)
    
    # set up model optimizer
    if optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f'Invalid optimizer {optimizer}')
    
    # set up scheduler for lr decay
    n_steps = n_epochs * len(train_loader)
    if scheduler == 'exp':
        min_lr_factor = lr_min / lr
        gamma = np.exp(np.log(min_lr_factor) / n_steps)
        scheduler = ExponentialLR(optimizer, gamma=gamma)
    elif scheduler == 'cos':
        scheduler = CosineAnnealingLR(optimizer, n_steps, eta_min=lr_min)
    else:
        raise ValueError(f'Invalid scheduler {scheduler}')
    
    # list to store results
    losses = list()

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0

        # standard NN training per batch
        for batch in train_loader:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            else:
                batch = batch

            optimizer.zero_grad()
            
            # estimate mean loss
            loss = - (model.log_prob(batch).sum()) / N  # N: [n obs.]
            # print(f"epoch loss nr. {epoch}.... loss {loss}************")

            # Backward pass
            loss.backward()

            total_norm = sum(p.grad.data.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5

            # updates the model's parameters
            optimizer.step()

            # sum losses for all bacthes
            epoch_loss += loss.cpu().item() / len(train_loader)    

            # update scheduler
            scheduler.step()

        losses.append(epoch_loss * N)
        # print(f"Epoch {epoch}: avg loss training: {loss:.4f},... Gradient Norm: {total_norm:.4f}")

    return model, losses



#######################################################################################################
#### Simple NSF PiecewiseRationalQuadraticCouplingTransform ###########################################
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.flows.base import Flow
from torch.nn import functional as F
from nflows.nn import nets as nets
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.normalization import BatchNorm

class SimpleNSF(Flow):
    def __init__(
        self,
        features,
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        num_bins=10,
        tails="linear",
        tail_bound=3.0,
        dropout_probability=0.0,
        activation=F.relu,
        apply_unconditional_transform = False,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
    ):
        mask = torch.ones(features)
        mask[::2] = -1

        def create_transform_net(in_features, out_features):
            return nets.ResidualNet(
                in_features,
                out_features,
                hidden_features=hidden_features,
                num_blocks=num_blocks_per_layer,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_layers,
            )

        layers = []
        for _ in range(num_layers):
            transform = PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=create_transform_net,
                num_bins=num_bins,
                tails=tails,
                tail_bound=tail_bound,
                apply_unconditional_transform = apply_unconditional_transform
            )
            layers.append(transform)
            mask *= -1
            if batch_norm_between_layers:
                layers.append(BatchNorm(features=features))

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal([features]),
        )


