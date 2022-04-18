from matplotlib import ticker
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import os

class Config():
    ''' Configs for our simulations'''
    
    def __init__(self, root_dir, exp, test_mode = False):
        ''' give root dir of project ('Inside Final Project') , experiment mode ('control', thresholding, changing_p), and whether it is test_mode (runs = runs vs runs = 3)'''
        valid = ['control', 'thresholding', 'changing_p']
        assert exp in valid
        
        # tickers
        self.tickers = ['AAPL', 'CVX', 'DVN', 'GS', 'JNJ', 'JPM', 'MRK', 'NVDA', 'PFE', 'TSLA', 'V', 'XOM']
        
        # directories
        self.root_dir = root_dir
        self.results_dir = os.path.join(root_dir, 'results')
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)
            
        self.test_dir = os.path.join(self.results_dir, exp)
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)
        
        self.sim1_dir = os.path.join(self.test_dir, 'sim1')
        self.sim2_dir = os.path.join(self.test_dir, 'sim2')
        if not os.path.exists(self.sim1_dir):
            os.mkdir(self.sim1_dir)
        if not os.path.exists(self.sim2_dir):
            os.mkdir(self.sim2_dir)
            
        self.metrics_dir = os.path.join(self.test_dir, 'metrics')
        if not os.path.exists(self.metrics_dir):
            os.mkdir(self.metrics_dir)
            
            
        # simulation1 parameters
        self.num_tests = 100
        if test_mode:
            self.num_tests = 3
        self.split = [.3, .7]
        self.pred_period = 140
        
        # simulation2 parameters
        self.page_rank_effect = (.95, .05)
        
        # get the training start point
        
    # Untested but should woirk
    def get_test_start_point(self, true_data):
        ''' This will be helpful during plotting to determine where the simulation data begins relative to true data (All data have the same size so this is universal)'''
        return int(len(true_data) * self.split[0])
    
    # Untested but should woirk
    def period_cutoffs(self, true_data):
        ''' This gives the period cutoffs. Will be useful for graphing. Includes start.
        Returns:
            Cutoffs by true_data index,
            Cutoffs by the actual index of data'''
        ind = true_data.index
        start = self.get_test_start_point(true_data)
        
        true_cutoffs = ind[start::self.pred_period]
        ind_cutoffs = list(range(len(true_data)))[start::self.pred_period]
        return true_cutoffs, ind_cutoffs
        
    def get_tickers(self,):
        ''' list of used tickers '''
        return self.tickers

    def load_sim1_data(self):
        """ Loads sim data beginning at path. Path should contain individual directories named after each stock"""
        # load simdata
        simdata = np.load(os.path.join(self.sim1_dir, f'sim1.npy'), allow_pickle=True).item()
        return simdata

    def load_sim2_data(self,):
        pageranks = np.load(os.path.join(self.sim2_dir, 'pageranks.npy'), allow_pickle=True).item()
        true_log_returns = np.load(os.path.join(self.sim2_dir, 'true_log_returns.npy'), allow_pickle=True).item()
        return pageranks, true_log_returns
    
    def load_metrics(self, top_n: int):
        metrics = np.load(os.path.join(self.results_dir, f'metrics_{top_n}.npy'), allow_pickle=True).item()
        return metrics
    
    def load_quantiles(self):
        quantiles = np.load(os.path.join(self.sim1_dir, 'quantiles.npy'), allow_pickle=True).item()
        return quantiles

    def load_true_data(self):
        """Pass the root directory"""
        # load data
        data = {}
        for file in os.listdir(os.path.join(self.root_dir,'data/clean_data')):
            print(file)
            ticker = file.split('.')[0] # retrieve ticker_name
            data[ticker] = pd.read_csv(filepath_or_buffer=os.path.join('data/clean_data/', file), header=0, index_col = 0, parse_dates=True, infer_datetime_format=True) # read data correctly
        return data

    
    #########################
    # Save Functions
    #########################
    
    def save_sim1(self, run_data):
        """ Saves Run data  (Aggregated over tickers) as an npy binary file"""
        np.save(os.path.join(self.sim1_dir, 'sim1.npy'), run_data, fix_imports=False)
        return

        
    def save_sim2(self, pageranks, true_log_returns):
        np.save(os.path.join(self.sim2_dir, 'pageranks.npy'), pageranks, fix_imports=False)
        np.save(os.path.join(self.sim2_dir, 'true_log_returns.npy'), true_log_returns, fix_imports=False)
        return

    def save_metrics(self, metrics, top_n : int ):
        np.save(os.path.join(self.metrics_dir, f'metrics_{top_n}.npy'), metrics, fix_imports = False)
        return

    def save_quantiles(self, quantiles):
        np.save(os.path.join(self.sim1_dir, 'quantiles.npy'), quantiles, fix_imports=False)
    



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
    
    
    
class PortfolioSim():
    
    ''' Computes a simulation for a portfolio w/ the Page Rank algorithm. '''
    def __init__(self,):
        pass

    def sim(self, ticker_sim1_data: dict, page_rank_effect = (.95, .05)):
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
        for period in trange(len(simdata[tickers[0]]), desc = 'Computing Page Ranks (Period)'):
            
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
                    
                means.append(mean_row)
                true_log_returns[ticker].append(np.sum(simdata[ticker][period,:,0, 1], axis = 0))
            means = np.array(means)
            
            # base array
            B = np.array([[1/len(tickers) for x in range(len(tickers))] for i in range(len(tickers))])
                
            # solve for limiting distribution
            G =  page_rank_effect[0] * means + page_rank_effect[1] * B
            # for each row, correct to a valid pdf
            for row in range(len(G)):
                G[row,:] /= G[row,:].sum()    
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
    



class Metrics():
    
    def __init__(self, sim1, page_ranks, true_log_returns, top_n:int,):
        ''' takes in simulation1 (dictionary form) and page_ranks, and true returns
        
        top_n = Number of stocks to buy each month'''
        self.sim1 = sim1
        self.page_ranks = page_ranks
        self.true_log_returns = true_log_returns
        
        self.top_n = top_n
    
    def all_metrics(self):
        metrics = {}
        proportion_purchased, tickers_purchased, pnl, ret = self.PnL()
        metrics['Proportion Purchased'] = proportion_purchased
        metrics['Tickers Purchased'] = tickers_purchased
        metrics['pnl'] = pnl
        metrics['ret'] = ret

        max_dd = self.max_drawdown()
        metrics['Max Drawdown'] = max_dd
        sharpe = self.Sharpe(0)
        metrics['Sharpe'] = sharpe

        return metrics
        
        
    
    def PnL(self):
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
        tickers = list(self.page_ranks.keys())
        for period in trange(len(self.page_ranks[tickers[0]]), desc = 'Computing PnL (period)'):
            period_page_rank = {}
            for ticker, pr in self.page_ranks.items():
                period_page_rank[pr[period]] = ticker
            
            top_scores = sorted(list(period_page_rank.keys()))[-self.top_n:]
            top_scores = np.array(top_scores)
            top_scores_tickers = [period_page_rank[i] for i in top_scores]
            top_scores = top_scores / top_scores.sum()
            proportion_purchased.append(top_scores)
            tickers_purchased.append(top_scores_tickers)
            
            cur_pnl = np.array([top_scores[x] * np.exp(self.true_log_returns[top_scores_tickers[x]][period]) for x in range(len(top_scores))]).sum()
            pnl.append(cur_pnl)
            
        ret = 1
        for cur_pnl in pnl:
            ret *= cur_pnl
        
        self.pnl = pnl
        self.ret = ret
        
        return proportion_purchased, tickers_purchased, pnl, ret
    
    # Untested completely (Should work though)
    def get_quantiles(self):
        ''' Returns a dict in similar format as the original sim 1.
        layer 1: Dict with (ticker, data)
        layer 2: Dict with ('quantile', data)
        layer 3-: It is simply a single run of 140 predictions for each period'''
        # Actual Max/Min simulation. For each period, we track the simulation with the highest/lowest ending value
        quantile_runs = {}
        tickers = self.sim1.keys()
        
        # for each ticker
        for ticker in tqdm(tickers, desc = 'Computing Quantiles (ticker)'):
            
            min_pos = []
            max_pos = []
            median = []
            q1 = []
            q3 = []
            
            num_sims = len(self.sim1[ticker])
            num_periods = len(self.sim1[ticker][0])
            
            for periodid in range(num_periods):
                temp = []
                for simid in range(num_sims):
                    temp.append(self.sim1[ticker][simid][periodid][:, 0].sum())
                temp = np.array(temp)
                min_pos.append(self.sim1[ticker][np.where(temp == temp.min())[0][0]][periodid])
                max_pos.append(self.sim1[ticker][np.where(temp == temp.max())[0][0]][periodid])
                median.append(self.sim1[ticker][np.where(temp == np.quantile(temp, .5, method = 'closest_observation'))[0][0]][periodid])
                q1.append(self.sim1[ticker][np.where(temp == np.quantile(temp, .25, method = 'closest_observation'))[0][0]][periodid])
                q3.append(self.sim1[ticker][np.where(temp == np.quantile(temp, .75, method = 'closest_observation'))[0][0]][periodid])

            quantile_runs[ticker] = {'Min':np.array(min_pos), 'Q1': np.array(q1), 'Median': np.array(median), 'Q3':(q3), 'Max': 
                np.array(max_pos)}
        return quantile_runs

    def max_drawdown(self,):
        ''' takes drawdown from PnL'''
        if self.pnl:
            pnl = np.array(self.pnl)
        else:
            pnl = np.array(self.PnL()[2])
        pnl -= 1 # to be returns
        return pnl.min()
        
                
            
    def Sharpe(self, rfrate = 0):
        """ Computes sharpe over time with unlogged data computed from PnL,
            Return a 2-d array of where:
                Outer Layer = 0 - runs # of simulations
                Layer 2 = 0 - # of test periods Sharpe calculations
                the first pnl will be NaN
            
        """
        sharpe = [np.nan]
        if self.pnl:
            pnl = np.array(self.pnl)
        else:
            pnl = np.array(self.PnL()[2])
        pnl -= 1
        for ret_index in range(1, len(pnl)):
            sharpe.append((pnl[:ret_index + 1].mean() - rfrate) / np.std(pnl[:ret_index+1]))
            
        self.sharpe = sharpe
        return np.array(sharpe)

        