import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from pathlib import Path


def results_kde(var, n_cv_splits = 5, kernel = 'gaussian', tune_bandwidths_cv = None, seed = 711):
    if tune_bandwidths_cv is None:
            tune_bandwidths_cv = np.logspace(-2, 1, 50)
    var = np.asarray(var).reshape(-1, 1)
    cv = KFold(n_splits=n_cv_splits, shuffle=True, random_state=seed)
    grid = GridSearchCV(KernelDensity(kernel=kernel),
                        {'bandwidth': tune_bandwidths_cv},
                        cv=cv)
    grid.fit(var)
    kde = grid.best_estimator_
    # print("Best bandwidth:", grid.best_params_['bandwidth'])
    log_dens = kde.score_samples(var)
    return np.sum(log_dens)/len(var), kde

def kde_density_curve(kde, data, grid_size=500):
    data = np.asarray(data)
    x_grid = np.linspace(data.min(), data.max(), grid_size).reshape(-1, 1)
    log_dens_grid = kde.score_samples(x_grid)
    dens_grid = np.exp(log_dens_grid)
    return x_grid.flatten(), dens_grid
