from utils import *
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from copy import deepcopy

# faster processing with ray
import ray

# Sample Code on Control
@ray.remote
class Changing_PSim(BaseSim): # how you inherit. (Now you have access to all of Base Sim's methods)
    
    def __init__(self):
        
        # define the states (the code should work with whatever state names you give it.)
        states = ['Bear', 'Bull']
        
        
        # Call the Base Sim's init method to init self.P, self.M, self.STD, self.states, self.strategies, and self.ret_colname
        super().__init__(states = states)
        # Any other attributes needed
        # None for this case
        
    ################
    # Override Functions
    ################
    
    # we need to overwrite 3 functions, defined below
    
    def compute_startstate(self, test_data: pd.DataFrame):
        ''' Returns start state'''
        # compute start state base on first values. Use ret_colname to index the correct return column
        first = test_data.iloc[0][self.ret_colname]
        
        # simple Bear bull
        if first >= 0:
            return 'Bull'
        else:
            return 'Bear'
    
    
    def init_train(self, train_data: pd.DataFrame):
        ''' This Train just recomputes probability over all time. No weights for nearer times. This is just the initial train'''
        # Compute self.P on the training data
        # use ret_colname
        
        # Use Base Sim reset method to set all means and P and std to 0
        self.reset() 
        
        # Make Base P
        self.allP = {day:{hour:self.new_prob() for hour in range(9,16)} for day in range(5)}
        # For self.STD, self.M, which is more involved computation. Initialize a dict of empty lists for each state
        rets = {day:{hour:dict(zip(self.states, [[] for i in range(len(self.states))])) for hour in range(9,16)} for day in range(5)}
        
        # Hold STD and M in allSTD and allM
        self.allSTD = {day:{hour:{} for hour in range(9,16)} for day in range(5)}
        self.allM = {day:{hour:{} for hour in range(9,16)} for day in range(5)}

        for rowid in range(len(train_data) - 1):
            cur_ret = train_data.iloc[rowid][self.ret_colname]
            next_ret = train_data.iloc[rowid + 1][self.ret_colname]
            
            cur_state = self.det_state(cur_ret) # helper method defined below
            next_state = self.det_state(next_ret)
            
            # Get time and day from data
            time_string = train_data.iloc[rowid].name
            day = time_string.day_of_week
            time = time_string.hour
            # just use self.P to keep track of counts first
            self.allP[day][time][cur_state][next_state] += 1
            # std/M is more involved
            rets[day][time][cur_state].append(cur_ret)
        
        # compute self.P, self.STD and self.M
        for day in range(5):
            for hour in range(9, 16):
                for state in self.states:
                    
                    # compute the totals for each row
                    state_total = 0
                    for next_state in self.states:
                        state_total += self.allP[day][hour][state][next_state]
                        
                    # compute the Probs
                    for next_state in self.states: 
                        self.allP[day][hour][state][next_state] = self.allP[day][hour][state][next_state]/state_total
                        
                    ret = np.array(rets[day][hour][state])
                    # compute self.M/self.STD
                    self.allM[day][hour][state] = ret.mean()
                    self.allSTD[day][hour][state] = ret.std()

        # return nothing
        return
    
    def retrain(self, last_month: pd.DataFrame):
        ''' This reTrain just recomputes probability over the last month, then averages it with the current data to form an exponential weighting of sorts. Most of the code is the same, just only on the last_month. The difference is the averaging at the end'''
        
        # Compute self.P on the training data
        # use ret_colname

        # do not reset data
        
        # For self.STD, self.M, which is more involved computation. Initialize a dict of empty lists for each state
        # Create a copy of each state !!!!!!!!!!!!!!
        
        # Make deep clones of the all statistics
        allP = {day:{hour:self.new_prob() for hour in range(9,16)} for day in range(5)}
        
        allM = deepcopy(self.allM)
        allSTD = deepcopy(self.allSTD)
        # Modify returns to hold returns based on day, hour combinations
        rets = {day:{hour:dict(zip(self.states, [[] for i in range(len(self.states))])) for hour in range(9,16)} for day in range(5)}
        
        for rowid in range(len(last_month) - 1):
            cur_ret = last_month.iloc[rowid][self.ret_colname]
            next_ret = last_month.iloc[rowid + 1][self.ret_colname]
            
            cur_state = self.det_state(cur_ret) # helper method defined below
            next_state = self.det_state(next_ret)
            # Get the day and hour
            time_string = last_month.iloc[rowid].name
            day = time_string.day_of_week
            time = time_string.hour
            # Use P to keep track of counts first
            allP[day][time][cur_state][next_state] += 1

            # std/M is more involved
            rets[day][time][cur_state].append(cur_ret)
            
        
        # compute P, STD, M
        for day in range(5):
            for hour in range(9, 16):
                for state in self.states:
                    
                    # compute the totals for each row
                    state_total = 0
                    for next_state in self.states:
                        state_total += allP[day][hour][state][next_state]
                    state_total = state_total if state_total != 0 else 1
                    # compute the Probs
                    for next_state in self.states:
                        allP[day][hour][state][next_state] = allP[day][hour][state][next_state]/state_total
                        
                    ret = np.array(rets[day][hour][state])
                    # compute self.M/self.STD
                    # Handle no new runs
                    allM[day][hour][state] = ret.mean() if len(ret) > 0 else allM[day][hour][state]
                    allSTD[day][hour][state] = ret.std() if len(ret) > 0 else allSTD[day][hour][state]
        
        # Now recompute self.P, self.STD, self.M
        # Compute per day, hour combination
        for day in range(5):
            for hour in range(9, 16):
                for state in self.states:
                    for next_state in self.states:
                        self.allP[day][hour][state][next_state] = (self.allP[day][hour][state][next_state] + allP[day][hour][state][next_state])/2
                        
                    self.allM[day][hour][state] = (self.allM[day][hour][state] + allM[day][hour][state])/2
                    self.allSTD[day][hour][state] = (self.allSTD[day][hour][state] + allSTD[day][hour][state])/2

        # return nothing
        return
    
    def test_step(self, train_data:pd.DataFrame, last_month: pd.DataFrame, test_data: pd.DataFrame, cur_time_step: int):
        # Nothing needs to be done in the test step for the control case, but you can adjust self.P and self.M and self.V
        time_string = test_data.iloc[cur_time_step].name
        day = time_string.day_of_week
        time = time_string.hour
        # time_string = test_data.iloc[cur_time_step]['timestamp'].split()
        # day = datetime.datetime.strptime(time_string, '%Y-%m-%d')
        # time = int(time_string[1][:2])
        # print(f'Changing to {day} {time}')
        # Change self.P, self.STD, self.M to right values for current state
        self.P = self.allP[day][time]
        self.M = self.allM[day][time]
        self.STD = self.allSTD[day][time]
        # print(self.M, self.STD)
        return
        
    # can write extra functions like this to be used in train
    def det_state(self, ret):
        if ret >= 0:
            return 'Bull'
        return 'Bear'       
        
if __name__ == '__main__':
    
    ray.init(include_dashboard = False)
    
    
    # set up dirs in results we will use (this will be changed)
    config = Config('.', 'changing_p', test_mode = True)
    
    
    # load data
    data = config.load_true_data()
    
    # send out ray multiprocess remote operation (Should not need to change this part)
    sims = []
    ticker_order = []
    for ticker, ohlc in tqdm(data.items(), desc = 'Ticker'):
        
        # send remote actor
        cp = Changing_PSim.remote()
        sim_data = cp.run_simulation.remote(runs = config.num_tests, data = ohlc, ret_colname = 'log_returns', split = config.split, pred_period = config.pred_period, drop_last_incomplete_period = True)
    
        ticker_order.append(ticker)
        sims.append(sim_data)

    # get data once it is ready 
    sims = ray.get(sims)
    all_sims = {}
    for ticker in range(len(ticker_order)):
        all_sims[ticker_order[ticker]] = sims[ticker] 
    
    # save simulation 1
    config.save_sim1(all_sims)
    ray.shutdown()

    # Run simulation 2
    ps = PortfolioSim()
    page_rank, true_log_returns = ps.sim(all_sims, config.page_rank_effect)
    
    # save simulation 2
    config.save_sim2(page_rank, true_log_returns)
    
    # compute quantiles and metrics for buying all stocks
    met = Metrics(all_sims, page_rank, true_log_returns, top_n = len(config.get_tickers()))
    quantiles = met.get_quantiles()
    config.save_quantiles(quantiles)
    
    metrics = met.all_metrics()
    config.save_metrics(metrics, top_n = len(config.get_tickers()))    