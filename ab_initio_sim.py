import numpy as np
from numpy.random import binomial, beta, poisson, triangular, exponential
from scipy.stats import binom, uniform
import pandas as pd
from random import shuffle

a=1

# Estimate of CTR for ad position on the serp
CTR = [7.06, 2.96, 2.1, 2.06, 1.48, 0.96, 0.9, 0.74, 0.52]

# Column name to column index, for working with numpy arrays without sacrificing column names
cols = {'bid': 0,  # bids of auction participants
        'theta': 1,  # angles of auction participants (to calculate search affinity)
        'qs': 2,  # static quality scores of participants
        'id': 3,  # to identify myself in the array
        'ad_rank': 4,  # search affinity * static quality score * bid
        'joins': 5,  # whether the ad rank is sufficient for taking part in auction
        'next_ad_rank': 6,  # ad rank of the next advertiser in the auction sorting
        'has_slot': 7,  # is there an ad slot available for the advertiser?
        'impression': 8,  # did the ad actually appear on the SERP?
        'click': 9,  # was the ad clicked?
        'cost': 10  # how much did the advertiser pay?
        }


def single_auction(kw_theta, my_infos, competitor_infos, slots, rank_threshold):
    """
    :param kw_theta: angle of the search keyword
    :param my_infos: np.array([[my_bid, my_theta, my_quality_score]])
    :param competitor_infos: np.array([[bid_1, theta_1, quality_score_1], ...])
    :param slots: [ctr_1, ..., ctr_num_slots]
    :param rank_threshold: minimum rank for the auction
    :return: impression, click, cost paid
    """
    # Merging my_infos and competitor_infos in one array:
    participants = np.zeros([competitor_infos.shape[0]+1, cols['cost']+1])
    participants[:, :competitor_infos.shape[1]] = np.concatenate([my_infos, competitor_infos])
    # Placing an identifier on my infos:
    participants[:, cols['id']] = np.array([1] + [0] * competitor_infos.shape[0])
    # Scalar product between keyword and adgroup for affinity:
    participants[:, cols['ad_rank']] = np.cos(participants[:, cols['theta']] - kw_theta)
    # Calculating ad rank (affinity times static quality score times bid):
    participants[:, cols['ad_rank']] *= participants[:, cols['qs']] * participants[:, cols['bid']]
    # Vector with 1 if rank is sufficient for taking part, 0 otherwise (np.round to avoid 0.5):
    participants[:, cols['joins']] = np.round((np.sign(participants[:, cols['ad_rank']] - rank_threshold) + 1) / 2)
    # Substituting the ad rank of non participating advertisers with the threshold, to use it as a next ad rank:
    # (no problem in doing that: in ranking they are at the end anyway)
    participants[:, cols['ad_rank']] *= participants[:, cols['joins']]
    participants[:, cols['ad_rank']] += (1 - participants[:, cols['joins']]) * rank_threshold
    # Sorting wrt ad rank:
    sorted_idx = np.argsort(participants[:, cols['ad_rank']])[::-1]
    participants = participants[sorted_idx]
    # For every advertiser, taking follower's ad rank in the auction:
    participants[:, cols['next_ad_rank']] = np.append(participants[1:, cols['ad_rank']], rank_threshold)
    # Column with info about ad slot availability:
    # (1 in the firts num_slots positions, 0 otherwise)
    num_slots = slots.shape[0]
    participants[:, cols['has_slot']] = 0
    participants[:num_slots, cols['has_slot']] = 1
    # Did the advertiser appear in the SERP?
    participants[:, cols['impression']] = participants[:, cols['has_slot']] * participants[:, cols['joins']]
    # Extracting random clicks:
    participants[:, cols['click']] = 0
    click = binomial(1, slots)[:participants.shape[0]]
    participants[:click.shape[0], cols['click']] = click
    participants[:, cols['click']] *= participants[:, cols['impression']]
    # Calculating the expense for the clicks:
    participants[:, cols['cost']] = participants[:, cols['click']] * participants[:, cols['next_ad_rank']]
    participants[:, cols['cost']] /= (participants[:, cols['ad_rank']] / participants[:, cols['bid']])
    # Calculating my infos:
    impression = (participants[:, cols['id']] * participants[:, cols['impression']]).sum()
    click = (participants[:, cols['id']] * participants[:, cols['click']]).sum()
    cost = (participants[:, cols['id']] * participants[:, cols['cost']]).sum()
    return int(impression), int(click), cost


def get_search_sample(mean, size=None):
    # How many searches today?
    return poisson(lam=mean, size=size)


def get_beta_parameters(mean, sigma):
    # Service function for translating mean, sigma parameters into alpha and beta parameters
    nu = mean * (1 - mean) / sigma**2 - 1
    alpha_ = mean * nu
    beta_ = (1 - mean) * nu
    return alpha_, beta_


def get_angle_sample(angle_std, size=None):
    # Extract a random angle for the advertiser
    assert angle_std < np.pi / 2
    std = angle_std / np.pi
    a, b = get_beta_parameters(1/2, std)
    sample = beta(a, b, size=size)
    sample -= 1/2
    sample *= np.pi
    return sample


def get_competitor_sample(mean, size=None):
    # How many competitors?
    return poisson(lam=mean, size=size)


def get_qs_sample(mode, size=None):
    # What is the static quality score of competitors?
    return np.round(triangular(1, mode, 10, size=size))


def get_bid_sample(mean, size=None):
    # What is the bid of competitors?
    return exponential(scale=mean, size=size)


def get_rank_threshold(angle_std, bid_mean, qs_mode, q):
    # To have a meaningful way to extract a rank threshold,
    # we select a quantile we want to exclude, and transform it into a rank threshold
    sample_size = 10000
    angle_sample = get_angle_sample(angle_std, sample_size)
    bid_sample = get_bid_sample(bid_mean, sample_size)
    qs_sample = get_qs_sample(qs_mode, sample_size)
    inst_qs_sample = np.cos(angle_sample) * bid_sample * qs_sample
    return np.quantile(inst_qs_sample, q)


def get_search_day(adgroup_parameters):
    """
    :param adgroup_parameters:
    :return: a list of searches for the day
    Each search is a dictionary that contains the following:
    'ad_group': key of the ad group (from adgroup_parameters) interested by the search
    'kw_theta', 'competitor_infos', 'slots', 'rank_threshold': see function single_auction
    """
    slots_cr = np.array(CTR)
    slots_cr /= 100

    # For every search parameter, generate a dictionary
    # where the key is the ad group key
    # For search-specific parameters, the value will be a list
    rank_threshold = {}
    searches = {}
    competitor_nums = {}
    search_kws = {}
    slots = {}

    for key in adgroup_parameters:
        # threshold for this adgroup for today:
        rank_threshold[key] = get_rank_threshold(adgroup_parameters[key]['angle_std'],
                                                 adgroup_parameters[key]['bid_mean'],
                                                 adgroup_parameters[key]['qs_mode'],
                                                 adgroup_parameters[key]['threshold_quantile'])
        # how many searches compatible with this adgroup today
        searches[key] = get_search_sample(adgroup_parameters[key]['search_mean'])
        # list of competitor numbers (one number for every search)
        competitor_nums[key] = get_competitor_sample(adgroup_parameters[key]['competitor_mean'], searches[key])
        # list of search keywords (represented by angles)
        search_kws[key] = get_angle_sample(adgroup_parameters[key]['angle_std'], size=searches[key])
        # slots for this adgroup
        slots[key] = slots_cr[:adgroup_parameters[key]['n_slots']]
        slots[key] = np.concatenate(
            [slots[key], np.array([0] * max(0, adgroup_parameters[key]['n_slots'] - slots[key].shape[0]))]
        )

    # for every ad group, for every search compatible with that ad group, create arguments for single_auction
    search_args = []
    for key in adgroup_parameters:
        for kw_theta, competitor_num in zip(search_kws[key], competitor_nums[key]):
            competitor_infos = np.zeros([competitor_num, 3])
            competitor_infos[:, 0] = get_bid_sample(adgroup_parameters[key]['bid_mean'], competitor_num)
            competitor_infos[:, 1] = get_angle_sample(adgroup_parameters[key]['angle_std'], competitor_num)
            competitor_infos[:, 2] = get_qs_sample(adgroup_parameters[key]['qs_mode'], competitor_num)
            search_args.append({'kw_theta': kw_theta, 'competitor_infos': competitor_infos,
                                'slots': slots[key], 'rank_threshold': rank_threshold[key], 'ad_group': key})
    # Mix searches compatible with the different ad groups
    shuffle(search_args)
    return search_args


def get_day_data(bids, budget, adgroup_parameters):
    """
    Get the result of one day of auctions for a campaign
    :param bids: the bids relative to all the ad groups of the campaign
    :param budget: budget of the day for the campaign
    :param adgroup_parameters:
    """
    # Initialize everything to zero
    impressions = {key: 0 for key in adgroup_parameters}
    clicks = {key: 0 for key in adgroup_parameters}
    cost = {key: 0 for key in adgroup_parameters}
    sat_impressions = {key: 0 for key in adgroup_parameters}
    sat_clicks = {key: 0 for key in adgroup_parameters}
    sat_cost = {key: 0 for key in adgroup_parameters}

    # Perform one auction for every search
    search_args = get_search_day(adgroup_parameters)
    for search in search_args:
        key = search['ad_group']
        my_bid = bids[key]
        my_qs = adgroup_parameters[key]['my_qs']
        search['my_infos'] = np.array([[my_bid, 0, my_qs]])  # centering my theta
        new_impressions, new_clicks, new_cost = single_auction(
            search['kw_theta'], search['my_infos'], search['competitor_infos'],
            search['slots'], search['rank_threshold']
        )
        sat_impressions[key] += new_impressions
        sat_clicks[key] += new_clicks
        sat_cost[key] += new_cost
        if budget >= 0:
            impressions[key] += new_impressions
            clicks[key] += new_clicks
            cost[key] += new_cost
            budget -= new_cost
    return sat_impressions, sat_clicks, sat_cost, impressions, clicks, cost


class CampaignSimulator:
    # Class recording the full history of simulated days
    def __init__(self, adgroup_parameters, lost_is_noise):
        self.adgroup_parameters = adgroup_parameters
        self.lost_is_noise = lost_is_noise

        self.bids = []
        self.budget = []

        self.sat_clicks = []
        self.sat_cpc = []
        self.sat_cost = []

        self.sat_impressions = []
        self.impressions = []

        self.act_clicks = []
        self.cost = []
        self.conversions = []
        self.lost_is = []
        self.inverse_cpc = []
        self.log_inverse_cpc = []
        self.inferred_clicks = []
        self.log_inferred_clicks = []

        self.censored_clicks = []

    def simulate_day(self, bids, budget):
        self.bids.append(bids)
        self.budget.append(budget)

        sat_impressions, sat_clicks, sat_cost, impressions, clicks, cost = get_day_data(bids, budget,
                                                                                        self.adgroup_parameters)

        self.sat_clicks.append(sat_clicks)
        self.sat_impressions.append(sat_impressions)
        self.sat_cost.append(sat_cost)
        self.sat_cpc.append({key: sat_cost[key] / sat_clicks[key] if sat_clicks[key] > 0 else np.nan for key in sat_cost})
        self.impressions.append(impressions)
        self.act_clicks.append(clicks)
        self.cost.append(cost)
        conversions = {key: binom.rvs(clicks[key], self.adgroup_parameters[key]['cr']) for key in clicks}
        self.conversions.append(conversions)

        lost_is = {}
        for key in sat_impressions:
            if sat_impressions[key] == 0:
                lost_is[key] = 0
            else:
                lost_is[key] = 1 - impressions[key] / sat_impressions[key]
            noisy_part = uniform.rvs()
            if noisy_part < 0.1:
                noisy_part = 0
            lost_is[key] = (1 - self.lost_is_noise) * lost_is[key] + self.lost_is_noise * noisy_part
        self.lost_is.append(lost_is)

        self.inverse_cpc.append({key: clicks[key] / cost[key] if cost[key] > 0 else np.nan for key in clicks})
        self.log_inverse_cpc.append({key: np.log(clicks[key] / cost[key]) if cost[key] * clicks[key] > 0 else np.nan for key in clicks})

        inferred_clicks = {key: clicks[key] / (1 - lost_is[key]) if lost_is[key] < 1 else 0 for key in lost_is}
        self.inferred_clicks.append(inferred_clicks)
        self.log_inferred_clicks.append({key: np.log(inferred_clicks[key]) if inferred_clicks[key] > 0 else np.nan for key in inferred_clicks})

        censored_clicks = {}
        for key in clicks:
            if clicks[key] < sat_clicks[key]:
                censored_clicks[key] = 1
            else:
                censored_clicks[key] = 0
        self.censored_clicks.append(censored_clicks)

    def produce_dfs(self):
        col_to_list = {'Impressions': self.impressions, 'Clicks': self.act_clicks, 'Cost': self.cost,
                       'Conversions': self.conversions,
                       'Budget': self.budget, 'Bid': self.bids, 'Search Lost IS (budget)': self.lost_is,
                       'Censored clicks': self.censored_clicks, 'Inverse CPC': self.inverse_cpc,
                       'Log Inverse CPC': self.log_inverse_cpc, 'Inferred Clicks': self.inferred_clicks,
                       'Log Inferred Clicks': self.log_inferred_clicks}
        dfs = {}
        # If at least one day has been simulated, generate df with the resulting values
        if len(self.act_clicks):
            adg_col_to_list = {}
            for col in col_to_list:
                for day in col_to_list[col]:
                    for adg in self.adgroup_parameters:
                        if adg not in adg_col_to_list:
                            adg_col_to_list[adg] = {}
                        if col not in adg_col_to_list[adg]:
                            adg_col_to_list[adg][col] = []
                        try:
                            adg_col_to_list[adg][col].append(day[adg])
                        except TypeError:
                            adg_col_to_list[adg][col].append(day)
            cols = [col for col in col_to_list]
            for adg in adg_col_to_list:
                df = pd.DataFrame(columns=cols)
                for col in cols:
                    df[col] = adg_col_to_list[adg][col]
                dfs[adg] = df
        # otherwise, generate row of nans
        else:
            for adg in self.adgroup_parameters:
                df = pd.DataFrame([[np.nan for col in col_to_list]], columns=[col for col in col_to_list])
                dfs[adg] = df

        return dfs


