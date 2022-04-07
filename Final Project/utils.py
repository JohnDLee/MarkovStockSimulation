import pandas as pd
import numpy as np
from tqdm import tqdm, trange

class BaseSim:
    
    def __init__(self, states: list):
        
        # set up states
        self.states = list(states)
        # initialize P
        self.P = dict(zip(list(states), [dict(zip(list(states), [0 for i in range(len(states))])) for i in range(len(states))]))
        self.M = dict(zip(list(states), [ 0 for i in range(len(states))]))
        self.STD = dict(zip(list(states), [ 0 for i in range(len(states))]))
        
        self.ret_colname = None
        self.close_colname = None
        self.strategies = ['B&H'] # only B&H for now
        
    def reset(self):
        ret_colname = self.ret_colname # preserve colname
        close_colname = self.close_colname
        self.__init__()
        self.ret_colname = ret_colname
        self.close_colname = close_colname 
        
    ####################
    # Simulation Fuction
    ####################
    
    def run_simulation(self, runs: int, data: pd.DataFrame, ret_colname = 'log_returns', close_colname = 'close',  split: list = [.5, .5], pred_period = 140, drop_last_incomplete_period = True):
        ''' Runs entire simulation and saves metrics
        Data Provided should already cleaned and ready to use. Must have a minimum of periodic returns as a column
        compute the start state of the simulation.
        
        
        Returns:
            Simulation data as a numpy, where each time period is [predicted, true].
            If drop_last_incomplete_period is true simulation data does not include the last time period if the data does not complete an entire prediction period'''
        
        self.ret_colname = ret_colname
        self.close_colname = close_colname
        
        # split data into train and test
        train_len = int(split[0] * len(data))
        train_data = data.iloc[:train_len-pred_period].copy()
        last_month = data.iloc[train_len - pred_period:train_len].copy()
        test_data = data.iloc[train_len:].copy()
        
        self.init_train(train_data)
        self.retrain(last_month)
        
        run_data = []
        # Monte Carlo Simulation
        for run in trange(runs, desc = 'Runs Completed'):
            
            sim = []
            testn = []
            cur_state = self.compute_startstate(test_data)
            # iterate through test data
            for time_step in range(len(test_data)):
                
                # every pred_period of data, add data to test data
                if time_step % pred_period == 0 and time_step != 0:
                    
                    # if it isn't first period, append the previous 140 and get the last month
                    train_data = pd.concat([train_data, last_month])
                    last_month = test_data.iloc[time_step-pred_period:time_step].copy()
                    
                    # save pred_periods worth of data into an array
                    sim.append(testn)

                    testn = [] # reset testn
                    
                    # retrain
                    self.retrain(last_month)
                    
                    
                # modifieable test_step
                self.test_step(train_data, last_month, test_data, time_step)
                
                # compute a return using normal distribution and save it w/ true value
                testn.append([np.random.normal(loc = self.M[cur_state], scale = self.STD[cur_state]), test_data.iloc[time_step][self.ret_colname], test_data.iloc[time_step][self.close_colname]] )
                # select the next state
                prob = np.random.uniform(0, 1)
                state_prob = 0
                for state in self.states:
                    state_prob += self.P[cur_state][state]
                    if prob < state_prob:
                        cur_state = state # set the next state
                        break
                    
            # ffill last value to match same length for numpy conversion
            if not drop_last_incomplete_period and pred_period - len(testn) != 0:
                testn += [testn[-1] for i in range(pred_period - len(testn))]
                # capture last piece of data
                sim.append(testn)
            
            # save to our runs
            run_data.append(sim) 
        
        
        return np.array(run_data)

    ###########################
    # Custom Functions to be defined in child classes
    ###########################
    
    def compute_startstate(self, test_data: pd.DataFrame):
        """Modify this step to compute the start state for the simulation. You are given the test_data. Ensure consistent column names when referencing or use self.ret_colname
        
        Return the start state"""
        pass
    
    def init_train(self, train_data: pd.DataFrame):
        """Modify this step to initially fill self.P with transition probs, self.M with mean returns, and self.STD with std of returns for each state.
        
        You are given the entire set of training data. Ensure when referencing training data, you use consistent column names or use self.ret_colname.
        
        Do not Return"""
        pass
    
    def retrain(self, last_month: pd.DataFrame):
        """Modify this step to continualy update self.P with transition probs, self.M with mean returns, and self.STD with std of returns for each state.
        
        You are given the entire set of training data. Ensure when referencing training data, you use consistent column names or use self.ret_colname.
        
        Do not Return"""
        pass
    
    def test_step(self, train_data:pd.DataFrame, last_month: pd.DataFrame, test_data: pd.DataFrame, cur_time_step: int):
        """Modify this test step to change P or perform other operations if necessary before computing return for that time period
        
        Do Not Return"""
        pass
    

    #########################
    # Statistic functions
    #########################
    def get_strategies(self):
        print('Valid Strategies:')
        for strat in self.strategies:
            print(f'\t{strat}')
    
    def compute_metrics(self, run_data: np.array, strategy = 'B&H', rfrate = 0):
        '''Should feed in simulation data after running. Will compute P/L (strategy), Sharpe, Loss
        Does not accept simulation data with Nans in it.
        run_data:
            Outer Layer = 0 - runs # of simulations
            Layer 2 = 0 - # of test periods
            Layer 3 = 0 - # of time periods per test period
            Layer 4 = [predicted, true],
            
        returns a dictionary of computed strategies'''
            
        pnl = self.PnL(run_data, strategy, log = True)
        sharpe = self.Sharpe( np.exp(pnl) -1, rfrate = rfrate)
        corr = self.corr(run_data)
        smape = self.SMAPE(run_data)
        
        return {'PnL': pnl, 'Sharpe': sharpe, 'Correlation': corr, 'SMAPE': smape}
        
        
        
    def PnL(self, run_data: np.array, strategy = 'B&H', log = True):
        """Compute P/L on run_data with strategy.
        
        Return a 2-d array of where:
            Outer Layer = 0 - runs # of simulations
            Layer 2 = 0 - # of test periods log P/L calculations
            """
        
        if strategy == 'B&H':
            pnl = []
            for sim in run_data:
                pnl_sim = []
                for period in sim:
                    expected_log_ret = period[:, 0].sum() # expected
                    actual_log_ret = period[:, 1].sum() # actual
                    
                    # check if profit or loss
                    # profit if actual matches expected dir
                    # loss if actual differs from expected
                    if actual_log_ret >= 0:
                        if expected_log_ret >= 0:
                            pnl_sim.append(actual_log_ret)
                        else:
                            pnl_sim.append(actual_log_ret * -1)
                    else:
                        if expected_log_ret < 0:
                            pnl_sim.append(actual_log_ret * -1)
                        else:
                            pnl_sim.append(actual_log_ret)
                            
                pnl.append(pnl_sim)
        pnl = np.array(pnl)
        if log:
            return pnl 
            
        return np.exp(pnl) - 1

    def Sharpe(self, PnL_data: np.array, rfrate = 0):
        """ Computes sharpe over time with unlogged data computed from PnL,
            Return a 2-d array of where:
                Outer Layer = 0 - runs # of simulations
                Layer 2 = 0 - # of test periods Sharpe calculations
                the first pnl will be NaN
            
        """
        sharpe = []
        for sim in PnL_data:
            sharpe_sim = [np.nan]
            for plindex in range(1, len(sim)):
                s = (sim[:plindex + 1].mean() - rfrate)/sim[:plindex+1].std()
                sharpe_sim.append(s)
            sharpe.append(sharpe_sim)
            
        return np.array(sharpe)
    
    def corr(self, run_data: np.array):
        """ Computes correlation of predicted model returns to true data returns
            Return a 2-d array of where:
                Outer Layer = 0 - runs # of simulations
                Layer 2 = 0 - # of test periods Corr calculations"""
        corrs = []
        for sim in run_data:
            corr_sim = []
            for period in sim:
                expected_log_ret = np.exp(period[:, 0]) - 1 # expected
                actual_log_ret = np.exp(period[:, 1]) - 1# actual
                corr_sim.append(np.corrcoef(expected_log_ret, actual_log_ret)[0][1])
            corrs.append(corr_sim)
        return np.array(corrs)

    def SMAPE(self, run_data: np.array):
        """ Computes correlation of predicted model to true data
            Return a 2-d array of where:
                Outer Layer = 0 - runs # of simulations
                Layer 2 = 0 - # of test periods SMAPE calculations"""
        smapes = []
        for sim in run_data:
            smape_sim = []
            for period in sim:
                expected_log_ret = np.exp(period[:, 0]) - 1 # expected
                actual_log_ret = np.exp(period[:, 1]) - 1# actual
                smape = 100/len(actual_log_ret) * np.sum(2 * np.abs(expected_log_ret - actual_log_ret) / (np.abs(actual_log_ret) + np.abs(expected_log_ret)))
                smape_sim.append(smape) 
            smapes.append(smape_sim)
        return np.array(smapes)
    

    #########################
    # Save Functions
    #########################
    
    def save_sim(self, run_data, filepath):
        """ Saves Run data as an npy binary file"""
        np.save(filepath, run_data, fix_imports=False)
        return
    
    def save_metrics(self, metrics:dict, filepath):
        """ Saves metrics in a npz zip file"""
        np.savez(filepath, **metrics)
        return
        