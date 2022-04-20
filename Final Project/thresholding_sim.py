
from utils import *
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# faster processing with ray
import ray

# Sample Code on Control
@ray.remote
class ThresholdingSim(BaseSim): # how you inherit. (Now you have access to all of Base Sim's methods)
    
    def __init__(self):
        
        # define the states (the code should work with whatever state names you give it.)
        self.percentages = [-0.005, -0.002, -0.001, 0.000, 0.001, 0.002, 0.005]
        states = [f'{"Bear" if percentage <= 0 else "Bull"} {percentage:.3f}' for percentage in self.percentages]
        states.append("Bull >0.005")
        # print(states)
        # Call the Base Sim's init method to init self.P, self.M, self.STD, self.states, self.strategies, and self.ret_colname
        super().__init__(states = states)
        
        # Any other attributes needed
        # None for this case
        
    ################
    # Override Functions
    ################
    
    # we need to overwrite 3 functions, defined below
    
    def compute_startstate(self, test_data: pd.DataFrame):
        """Modify this step to compute the start state for the simulation. You are given the test_data. Ensure consistent column names when referencing or use self.ret_colname
        
        Return the start state"""
        first = test_data.iloc[0][self.ret_colname]

        return self.det_state(first)
    
    
    def init_train(self, train_data: pd.DataFrame):
        """Modify this step to initially fill self.P with transition probs, self.M with mean returns, and self.STD with std of returns for each state.
        
        You are given the entire set of training data. Ensure when referencing training data, you use consistent column names or use self.ret_colname.
        
        Do not Return"""

        self.reset() 

        rets = dict(zip(self.states, [[] for i in range(len(self.states))]))

        for rowid in range(len(train_data) - 1):
            cur_ret = train_data.iloc[rowid][self.ret_colname]
            next_ret = train_data.iloc[rowid + 1][self.ret_colname]

            
            # Helpers defined above
            cur_state = self.det_state(cur_ret) 
            next_state = self.det_state(next_ret)
            
            # use P to keep track of counts first
            self.P[cur_state][next_state] += 1

            # std/M is more involved
            rets[cur_state].append(cur_ret)
        

        # compute self.P, self.STD and self.M
        for state in self.states:
            
            # compute the totals for each row
            state_total = 0
            for next_state in self.states:
                # print(self.P)
                state_total += self.P[state][next_state]
                
            # compute the Probs
            for next_state in self.states:
                self.P[state][next_state] = self.P[state][next_state] / state_total
                
            ret = np.array(rets[state])
            # compute self.M/self.STD
            self.M[state] = ret.mean()
            self.STD[state] = ret.std()
        # print(self.P)
        return
    
    
    def retrain(self, last_month: pd.DataFrame):
        """Modify this step to continualy update self.P with transition probs, self.M with mean returns, and self.STD with std of returns for each state.
        
        You are given the entire set of training data. Ensure when referencing training data, you use consistent column names or use self.ret_colname.
        
        Do not Return"""
        P = self.P.copy()
        M = self.M.copy()
        STD = self.STD.copy()
        rets = dict(zip(self.states, [[] for i in range(len(self.states))]))

        # Compute probability matrix based on last month
        for rowid in range(len(last_month) - 1):
            cur_ret = last_month.iloc[rowid][self.ret_colname]
            next_ret = last_month.iloc[rowid + 1][self.ret_colname]
            
            cur_state = self.det_state(cur_ret) # helper method defined below
            next_state = self.det_state(next_ret)
            
            # use P to keep track of counts first
            P[cur_state][next_state] += 1
            # std/M is more involved
            rets[cur_state].append(cur_ret)

        # Compute Statistics
        for rowid in range(len(last_month) - 1):
            cur_ret = last_month.iloc[rowid][self.ret_colname]
            next_ret = last_month.iloc[rowid + 1][self.ret_colname]
            
            cur_state = self.det_state(cur_ret) # helper method defined below
            next_state = self.det_state(next_ret)
            
            # use P to keep track of counts first
            P[cur_state][next_state] += 1
            # std/M is more involved
            rets[cur_state].append(cur_ret) 

        # Now recompute self.P, self.STD, self.M
        for state in self.states:
            rowSum = int(sum(P[state].values()))
            # print(rowSum, self.P[state])
            # print(rowSum, [P[state][s] / rowSum for s in P[state]])
            for next_state in self.states:
                # print(self.P[state][next_state])
                self.P[state][next_state] = (self.P[state][next_state] / rowSum + (P[state][next_state] / rowSum))/2
                
            self.M[state] = (self.M[state] + M[state])/2
            self.STD[state] = (self.STD[state] + STD[state])/2
            # print(f'{state}: {self.M[state]} {self.STD[state]}')
        # print(self.P)
        return
    
        
    def det_state(self, ret):
        '''Computes state of return, returns state'''
        # Populate states
        states = self.percentages
        
        # Remove the greater than
        # states.pop()
        # print(ret)
        # Determine current state
        for state in states:
            if ret < state:
                # print(f'{"Bear " if ret < 0 else "Bull "}{state:.3f}')
                return f'{"Bear " if ret < 0 else "Bull "}{state:.3f}'
        # print('Bull >0.005')
        return 'Bull >0.005'
        
if __name__ == '__main__':
    
    ray.init(include_dashboard = False)
    
    # set up dirs in results we will use (this will be changed)
    config = Config('.', 'thresholding', test_mode = False)
    
    
    # load data
    data = config.load_true_data()

    # send out ray multiprocess remote operation (Should not need to change this part)
    sims = []
    ticker_order = []
    for ticker, ohlc in tqdm(data.items(), desc = 'Ticker'):
        
        # send remote actor
        t = ThresholdingSim.remote()
        sim_data = t.run_simulation.remote(runs = config.num_tests, data = ohlc, ret_colname = 'log_returns', split = config.split, pred_period = config.pred_period, drop_last_incomplete_period = True)
    
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
    

                

        

    

