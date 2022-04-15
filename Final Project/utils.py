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
            Simulation data as a numpy, where each time period is [predicted_ret, true_ret, true_close].
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
    # Save Functions
    #########################
    
    def save_sim(self, run_data, filepath):
        """ Saves Run data as an npy binary file"""
        np.save(filepath, run_data, fix_imports=False)
        return
    
    
    
    ##########################
    # Deprecated - Do Not Use
    ########################## 
    
    def save_metrics(self, metrics:dict, filepath):
        """ Saves metrics in a npz zip file"""
        np.savez(filepath, **metrics)
        return
    
    ##########################
    # Statistic functions
    ##########################
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
    
    
    
class BasePortfolioSim():
    
    ''' Computes a simulation for a portfolio w/ the Page Rank algorithm. '''
    def __init__(self,):
        pass

    
    def sim(self, ticker_sim1_data: dict):
        ''' allticker_sim1_data should be in form:
                Outer Layer = Tickers (0-num_tickers)
                Layer 2 = 0 - runs # of simulations
                Layer 3 = 0 - # of test periods
                Layer 4 = 0 - # of time periods per test period
                Layer 5 = [predicted_ret, true_ret, true_close]
                
        '''
        
        tickers = list(ticker_sim1_data.keys())
        
        # simulation results
        pageranks = dict(zip(tickers, [[] for x in range(len(tickers))])) # page ranks at each month for each stock
        true_log_returns = dict(zip(tickers, [[] for x in range(len(tickers))])) # true returns for each month for each stock
        
        
        
        simdata = {}
        # First, swap the axis around to preserve data
        for ticker in tickers:
            simdata[ticker] = np.swapaxes(ticker_sim1_data[ticker], 0, 1)  
            simdata[ticker] = np.swapaxes(simdata[ticker], 1, 2)  

        # for each period
        for period in range(len(simdata[tickers[0]])):
            
            # compute the positive PDF for each stock.
            means = []
            for ticker in tickers:
                simulation_ret = np.exp(np.sum(simdata[ticker][period,:,:, 0], axis = 0))
                mean_row = []
                for ticker2 in tickers:
                    
                    # if they are the same, give it a 0 weight
                    if ticker2 == ticker:
                        mean_row.append(0)
                        continue
                    
                    # otherwise compute density of area under curve above 0
                    sim2_ret_mean = np.exp(np.sum(simdata[ticker2][period,:,:, 0], axis = 0))
                    data = simulation_ret - sim2_ret_mean
                    weight, bins = np.histogram(data, bins = np.linspace(data.min(), data.max(), 100), density = True)
                    mean_row.append(weight[bins[:-1] > 0].sum())
                mean_row = np.array(mean_row)
                # normalize to a probability
                mean_row /= mean_row.sum()
                means.append(mean_row)
                true_log_returns[ticker].append(np.sum(simdata[ticker][period,:,0, 1], axis = 0))
            means = np.array(means)
            
            # base array
            B = np.array([[1/len(tickers) for x in range(len(tickers))] for i in range(len(tickers))])
                
            # solve for limiting distribution
            G = .75 * means + .25 * B
            G = np.transpose(G) 
            G = G - np.identity(len(tickers))
            G[-1,:] = 1
            solutions = [0 for x in range(len(tickers))]
            solutions[-1] = 1
            
            pr = np.linalg.solve(G, solutions)
            # append it to page_ranks in correct order
            for ticker_id in range(len(tickers)):
                pageranks[tickers[ticker_id]].append(pr[ticker_id])
            

        return pageranks, true_log_returns
       


def get_tickers():
    ''' list of used tickers '''
    return ['AAPL', 'CVX', 'DVN', 'GS', 'JNJ', 'JPM', 'MRK', 'NVDA', 'PFE', 'TSLA', 'V', 'XOM']

def load_sim_data(path, tickers):
    """ Loads sim data beginning at path. Path should contain individual directories named after each stock"""
    # load simdata
    simdata = {}
    for ticker in tickers:
        simdata[ticker] = np.load(os.path.join(path, f'{ticker}/simulation.npy'))
    return simdata

def load_true_data(root_dir):
    """Pass the root directory"""
    # load data
    data = {}
    for file in os.listdir('data/clean_data'):
        print(file)
        ticker = file.split('.')[0] # retrieve ticker_name
        data[ticker] = pd.read_csv(filepath_or_buffer=os.path.join('data/clean_data/', file), header=0, index_col = 0, parse_dates=True, infer_datetime_format=True) # read data correctly
    return data


                
def PnL(portfolio_page_rank, portfolio_true_ret, choose: int):
    ''' Takes results from simulation and outputs PnL,
    Chooses "choose" number of stocks to purchase at each time period
    Changes from log returns to normal returns.
    
    returns:
        list of proportions of each stock purchased
        list of each stock purchased
        list of pnl's 
        final pnl after all time.'''
    
    proportion_purchased = []
    tickers_purchased = []
    pnl = []
    tickers = list(portfolio_page_rank.keys())
    for period in range(len(portfolio_page_rank[tickers[0]])):
        period_page_rank = {}
        for ticker, pr in portfolio_page_rank.items():
            period_page_rank[pr[period]] = ticker
        
        top_scores = sorted(list(period_page_rank.keys()))[-choose:]
        top_scores = np.array(top_scores)
        top_scores_tickers = [period_page_rank[i] for i in top_scores]
        top_scores = top_scores / top_scores.sum()
        proportion_purchased.append(top_scores)
        tickers_purchased.append(top_scores_tickers)
        
        cur_pnl = np.array([top_scores[x] * np.exp(portfolio_true_ret[top_scores_tickers[x]][period]) for x in range(len(top_scores))]).sum()
        pnl.append(cur_pnl)
        
    ret = 1
    for cur_pnl in pnl:
        ret *= cur_pnl
    
    return proportion_purchased, tickers_purchased, pnl, ret
        
    
def max_drawdown(pnl):
    ''' takes pnl from PnL'''
    pnl = np.array(pnl)
    pnl -= 1 # to be returns
    return pnl.min()
    
            
        
def Sharpe( pnl: np.array, rfrate = 0):
    """ Computes sharpe over time with unlogged data computed from PnL,
        Return a 2-d array of where:
            Outer Layer = 0 - runs # of simulations
            Layer 2 = 0 - # of test periods Sharpe calculations
            the first pnl will be NaN
        
    """
    sharpe = [np.nan]
    pnl = np.array(pnl)
    pnl -= 1
    for ret_index in range(1, len(pnl)):
        sharpe.append((pnl[:ret_index + 1].mean() - rfrate) / np.std(pnl[:ret_index+1]))
    return np.array(sharpe)   
    
    
    
    
            
        

        
        
# testing
if __name__ == '__main__':
    import os
    
    tickers = get_tickers()
    root = 'results/Control'
    
    # load simdata
    simdata = load_sim_data(root, tickers)
    
    test= BasePortfolioSim()
    pageranks, true_log_returns = test.sim(simdata)
        
            
    prop, tic, pnl, ret = PnL(pageranks, true_log_returns, choose = len(tickers))
    print(ret)
    print(Sharpe(pnl, rfrate=0))
    print(max_drawdown(pnl))