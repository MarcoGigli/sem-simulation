import os
from time import time
import pickle
import pandas as pd


def baseline_choice(day_report, total_budget, min_bid, max_bid):
    """
    Baseline function that 
    - shares the total_budget equally among campaigns
    - sets as bid the midpoint of [min_bid, max_bid]
    ignoring the variable day_report
    """
    campaigns_num = len(day_report['campaign_id'].unique())
    budgets = {campaign_id: total_budget / campaigns_num for campaign_id in list(day_report['campaign_id'].unique())}
    bids = {}
    for campaign_id in budgets.keys():
        ad_group_ids = list(day_report[day_report['campaign_id'] == campaign_id]['ad_group_id'].unique())
        bids[campaign_id] = {ad_group_id: .5 * (min_bid + max_bid) for ad_group_id in ad_group_ids}
        
    return budgets, bids


def simulate(sim, choice_function, total_budget, num_rounds, min_bid, max_bid, result_folder, result_name):
    """
    :param sim: 
    :param choice_function: f: (day_report, total_budget, min_bid, max_bid) -> budgets, bids
    :param total_budget: to be split every day among campaigns
    :param num_rounds: number of simulation days
    :param min_bid: minimum admitted bid
    :param max_bid: maximum admitted bid
    :param result_folder: where we want to save the pickle
    :param result_name: file name will be result_name.pkl
      
    :return: sim
    """
    if not os.path.exists(result_folder):
            os.mkdir(result_folder)
    
    for day in range(num_rounds):
        start = time()
        
        report_list = []
        for campaign_id, campaign_sim in sim.items():
            adgroup_reports = campaign_sim.produce_dfs()
            for adgroup_id, adgroup_report in adgroup_reports.items():
                adgroup_report['campaign_id'] = campaign_id
                adgroup_report['ad_group_id'] = adgroup_id
                adgroup_report['Min Bid'] = min_bid
                adgroup_report['Max Bid'] = max_bid
                report_list.append(adgroup_report)
        day_report = pd.concat(report_list)
        
        new_budgets, new_bids = choice_function(day_report, total_budget, min_bid, max_bid)
            
        for campaign_id, budget in new_budgets.items():
            bids = new_bids[campaign_id]
            sim[campaign_id].simulate_day(bids, budget)
        
        with open(os.path.join(result_folder, result_name + '.pkl'), 'wb') as file:
            pickle.dump(sim, file)
            
        print('*****', result_name, day, time() - start, '*****')
        
    return sim


def get_total_conversions(sim):
    """
    Returns an array (one entry per day of simulation) with the total 
    conversions of the day (summed over campaigns)
    """
    conversions = 0
    for campaign_id, campaign_sim in sim.items():
        dfs = campaign_sim.produce_dfs()
        for ad_group_id, df in dfs.items():
#             print(len(df))
            conversions += df['Conversions'].values
    return conversions


def get_regret(sim, reference_sim):
    """
    Regret suffered in sim due to not adopting strategy of 
    reference_sim
    """
    sim_tot_conv = get_total_conversions(sim)
    ref_tot_conv = get_total_conversions(reference_sim)
    # If simulation lengths differ, we take the minimum
    l = min(len(sim_tot_conv), len(ref_tot_conv))
    regret = sim_tot_conv[:l] - ref_tot_conv[:l]
    regret = pd.Series(regret).cumsum()
    relative_regret = regret / pd.Series(ref_tot_conv[:l]).cumsum()
    return regret.values, relative_regret.values