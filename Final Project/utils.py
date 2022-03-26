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
        self.strategies = ['B&H'] # only B&H for now
        
    def reset(self):
        ret_colname = self.ret_colname # preserve colname
        self.__init__(self.states)
        self.ret_colname = ret_colname 
        
    ####################
    # Simulation Fuction
    ####################
    
    def run_simulation(self, runs: int, data: pd.DataFrame, ret_colname = 'returns', split: list = [.5, .5], pred_period = 140):
        ''' Runs entire simulation and saves metrics
        Data Provided should already cleaned and ready to use. Must have a minimum of periodic returns as a column
        compute the start state of the simulation.
        
        Returns:
            Simulation data as a numpy, where each time period is [predicted, true]'''
        
        self.ret_colname = ret_colname
        
        # split data into train and test
        train_len = split[0] * len(data)
        train_data = data[:train_len].copy()
        test_data = data[train_len:].copy()
        
        self.train(train_data)
        
        runs = []
        # Monte Carlo Simulation
        for run in trange(runs, desc = 'Runs Completed'):
            
            sim = []
            testn = []
            cur_state = self.compute_startstate(test_data)
            # iterate through test data
            for time_step in trange(len(test_data), desc = 'Timestep'):
                
                # every pred_period of data, add data to test data
                if time_step % pred_period == 0 and time_step != 0:
                    
                    # if it isn't first period, append the previous 140 and remove from test
                    train_data.append(test_data[time_step-pred_period:time_step].copy())
                    
                    # save pred_periods worth of data into an array
                    sim.append(testn)
                    testn = [] # reset testn
                    
                    # retrain
                    self.train(train_data)
                    
                    
                # modifieable test_step
                self.test_step(train_data, test_data)
                
                # compute a return using normal distribution and save it w/ true value
                testn.append([np.random.normal(loc = self.M[cur_state], scale = self.STD[cur_state]), test_data[time_step][self.ret_colname]] )
                
                # select the next state
                prob = np.random.uniform(0, 1)
                state_prob = 0
                for state in self.states:
                    state_prob += self.P[cur_state][state]
                    if prob < state_prob:
                        cur_state = state # set the next state
                        break
            # capture last piece of data
            sim.append(testn)
            # save to our runs
            runs.append(sim) 
        
        
        return np.array(runs)

    ###########################
    # Custom Functions to be defined in child classes
    ###########################
    
    def compute_startstate(self, test_data: pd.DataFrame):
        """Modify this step to compute the start state for the simulation. You are given the test_data. Ensure consistent column names when referencing or use self.ret_colname
        
        Return the start state"""
        pass
    
    def train(self, train_data: pd.DataFrame):
        """Modify this step to initially fill self.P with transition probs, self.M with mean returns, and self.STD with std of returns for each state.
        
        You are given the entire set of training data. Ensure when referencing training data, you use consistent column names or use self.ret_colname.
        
        Do not Return"""
        pass
    
    def test_step(self, train_data:pd.DataFrame, test_data: pd.DataFrame):
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
    
    def compute_metrics(self, run_data: np.array, strategy = 'B&H'):
        '''Should feed in simulation data after running. Will compute P/L (strategy), Sharpe, Loss
        run_data:
            Outer Layer = 0 - runs # of simulations
            Layer 2 = 0 - # of test periods
            Layer 3 = 0 - # of time periods per test period
            Layer 4 = [predicted, true],
            
        returns a dictionary of computed strategies'''
            
        pnl = self.PnL(run_data, strategy)
        sharpe = self.Sharpe(run_data, pnl)
        corr = self.corr(run_data)
        smape = self.SMAPE(run_data)
        
        return {'PnL': pnl, 'Sharpe': sharpe, 'Correlation': corr, 'SMAPE': smape}
        
        
        
    def PnL(self, run_data: np.array, strategy = 'B&H'):
        """Compute P/L on run_data with strategy.
        
        Return a 2-d array of where:
            Outer Layer = 0 - runs # of simulations
            Layer 2 = 0 - # of test periods P/L calculations
            """
        return

    def Sharpe(self, run_data: np.array, PnL_data: np.array):
        """ Computes sharpe using run_data and data computed from PnL,
            Return a 2-d array of where:
                Outer Layer = 0 - runs # of simulations
                Layer 2 = 0 - # of test periods Sharpe calculations
            
        """
        return
    
    def corr(self, run_data: np.array):
        """ Computes correlation of predicted model to true data
            Return a 2-d array of where:
                Outer Layer = 0 - runs # of simulations
                Layer 2 = 0 - # of test periods Corr calculations"""
        return

    def SMAPE(self, run_data: np.array):
        """ Computes correlation of predicted model to true data
            Return a 2-d array of where:
                Outer Layer = 0 - runs # of simulations
                Layer 2 = 0 - # of test periods SMAPE calculations"""
        return
    

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
        