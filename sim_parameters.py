import numpy as np

base_parameters = {
    'my_qs': 9, 'my_budget': np.inf,
    'search_mean': 10000,
    'angle_std': np.pi/10,
    'competitor_mean': 10, 'my_mode': 7, 'bid_mean': .3,
    'threshold_quantile': .3, 'n_slots': 2
}

parameter_ranges = {
    'my_qs': [8, 9, 10],
    'search_mean': [1000, 2000, 5000, 6000, 7000, 8000, 10000],
    'angle_std': np.pi / np.array([20, 15, 10, 5]),
    'competitor_mean': [2, 5, 6, 8, 10],
    'qs_mode': [6, 7, 8, 9],
    'bid_mean': [.1, .2, .3, .4],
    'threshold_quantile': [.01, .1, .2, .3],
    'n_slots': [2, 4, 6, 8],
    'cr': [0.008, 0.01, 0.015, 0.02]
}

hyperparameter_ranges = {
    'n_campaigns': [2, 4, 8],
    'n_adgroups_per_campaign': [1, 2, 4],
    'lost_is_noise': [.01, .1, .2, .5],
}