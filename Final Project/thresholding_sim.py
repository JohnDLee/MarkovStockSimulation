from msilib.schema import Control
from utils import BaseSim
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
        percentages = [-0.05, -0.02, -0.01, 0.00, 0.01, 0.02, 0.05]
        states = [f'{"Bear" if percentage <= 0 else "Bull"} {percentage}' for percentage in percentages].append('Bull >0.05')
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

        return self.ret_state(first)
    
    
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
                state_total += self.P[state][next_state]
                
            # compute the Probs
            for next_state in self.states:
                self.P[state][next_state] = self.P[state][next_state]/state_total
                
            ret = np.array(rets[state])
            # compute self.M/self.STD
            self.M[state] = ret.mean()
            self.STD[state] = ret.std()

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
            for next_state in self.states:
                self.P[state][next_state] = (self.P[state][next_state] + P[state][next_state])/2
                
            self.M[state] = (self.M[state] + M[state])/2
            self.STD[state] = (self.STD[state] + STD[state])/2
        
        return
    
    def test_step(self, train_data: pd.DataFrame,  test_data: pd.DataFrame):
        # Nothing needs to be done in the test step for the control case, but you can adjust self.P and self.M and self.V
        return
        
    def det_state(self, ret):
        '''Computes state of return, returns state'''
        # Populate states
        states = [float(state[-4:]) for state in self.states]
        
        # Remove the greater than
        states.pop()

        # Determine current state
        for state in states:
            if ret < state:
                return f'{"Bear" if ret < 0 else "Bull"} {state}'
        return 'Bull >0.05'
        
if __name__ == '__main__':
    
    ray.init(include_dashboard = False)
    
    # set up dirs in results we will use (this will be changed)
    results_dir = 'results/Control'
    os.system(f'rm -rf {results_dir}')
    os.mkdir(results_dir)
    
    
    # load data
    data = {}
    for file in os.listdir('data/clean_data'):
        print(file)
        ticker = file.split('.')[0] # retrieve ticker_name
        data[ticker] = pd.read_csv(filepath_or_buffer=os.path.join('data/clean_data/', file), header=0, index_col = 0, parse_dates=True, infer_datetime_format=True) # read data correctly
    

    runs = 100 # for testing
    # send out ray multiprocess remote operation (Should not need to change this part)
    sims = []
    metrics = []
    ticker_order = []
    for ticker, ohlc in tqdm(data.items(), desc = 'Ticker'):
        
        # create dir
        ticker_path = os.path.join(results_dir, ticker)
        os.mkdir(ticker_path)
        
        # send remote actor
        threshold = Control.remote()
        sim_data = threshold.run_simulation.remote(runs = runs, data = ohlc, ret_colname = 'log_returns', split = [.5, .5], pred_period = 140, drop_last_incomplete_period = True) # 140 for a month
        metric_data = threshold.compute_metrics.remote(sim_data)
        

        # save the data in results
        threshold.save_sim.remote(sim_data, os.path.join(ticker_path, 'simulation.npy'))
        threshold.save_metrics.remote(metric_data, os.path.join(ticker_path, 'metrics.npz'))
        
        ticker_order.append(ticker)
        sims.append(sim_data)
        metrics.append(metric_data)
    
    # get data once it is ready 
    sims = ray.get(sims)
    metrics = ray.get(metrics)
    
    #print(sims[0])
    ray.shutdown()
    



                

        

    

