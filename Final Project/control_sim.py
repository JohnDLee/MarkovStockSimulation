from utils import BaseSim
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# faster processing with ray
import ray

# Sample Code on Control
@ray.remote
class ControlSim(BaseSim): # how you inherit. (Now you have access to all of Base Sim's methods)
    
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
        
        # For self.STD, self.M, which is more involved computation. Initialize a dict of empty lists for each state
        rets = dict(zip(self.states, [[] for i in range(len(self.states))]))
        
        for rowid in range(len(train_data) - 1):
            cur_ret = train_data.iloc[rowid][self.ret_colname]
            next_ret = train_data.iloc[rowid + 1][self.ret_colname]
            
            cur_state = self.det_state(cur_ret) # helper method defined below
            next_state = self.det_state(next_ret)
            
            # just use self.P to keep track of counts first
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
        
        # return nothing
        return
    
    def retrain(self, last_month: pd.DataFrame):
        ''' This reTrain just recomputes probability over the last month, then averages it with the current data to form an exponential weighting of sorts. Most of the code is the same, just only on the last_month. The difference is the averaging at the end'''
        
        # Compute self.P on the training data
        # use ret_colname

        # do not reset data
        
        # For self.STD, self.M, which is more involved computation. Initialize a dict of empty lists for each state
        # Create a copy of each state !!!!!!!!!!!!!!
        P = self.P.copy()
        M = self.M.copy()
        STD = self.STD.copy()
        rets = dict(zip(self.states, [[] for i in range(len(self.states))]))
        
        for rowid in range(len(last_month) - 1):
            cur_ret = last_month.iloc[rowid][self.ret_colname]
            next_ret = last_month.iloc[rowid + 1][self.ret_colname]
            
            cur_state = self.det_state(cur_ret) # helper method defined below
            next_state = self.det_state(next_ret)
            
            # use P to keep track of counts first
            P[cur_state][next_state] += 1
            # std/M is more involved
            rets[cur_state].append(cur_ret)
        
        # compute P, STD, M
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
            M[state] = ret.mean()
            STD[state] = ret.std()
        
        # Now recompute self.P, self.STD, self.M
        for state in self.states:
            for next_state in self.states:
                self.P[state][next_state] = (self.P[state][next_state] + P[state][next_state])/2
                
            self.M[state] = (self.M[state] + M[state])/2
            self.STD[state] = (self.STD[state] + STD[state])/2
            
        # return nothing
        return
    
    # can write extra functions like this to be used in train
    def det_state(self, ret):
        if ret >= 0:
            return 'Bull'
        return 'Bear'       
        
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
    

    runs = 10 # for testing
    # send out ray multiprocess remote operation (Should not need to change this part)
    sims = []
    ticker_order = []
    for ticker, ohlc in tqdm(data.items(), desc = 'Ticker'):
        
        # create dir
        ticker_path = os.path.join(results_dir, ticker)
        os.mkdir(ticker_path)
        
        # send remote actor
        control = ControlSim.remote()
        sim_data = control.run_simulation.remote(runs = runs, data = ohlc, ret_colname = 'log_returns', split = [.75, .25], pred_period = 140, drop_last_incomplete_period = True) # 140 for a month

        # save the data in results
        control.save_sim.remote(sim_data, os.path.join(ticker_path, 'simulation.npy'))
        
        ticker_order.append(ticker)
        sims.append(sim_data)

    # get data once it is ready 
    sims = ray.get(sims)
    
    #print(sims[0])
    ray.shutdown()
    
