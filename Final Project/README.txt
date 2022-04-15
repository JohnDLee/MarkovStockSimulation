install requirements by running

% pip install -r requirements.txt

Run retrieve_data.py after to retrieve data into a data folder



Notes for Project

Steps:
    1. Retrieve/Store Data
        - ID/Secretkey
            - PKF1UA7IOK2GNIOZJCCE
            - qoUq6tjPOwpwDld6DfjTvov5q0bAj0KvvHnUSUDx
        - Stocks
            - Tech
                - TSLA, AAPL, NVDA
            - Finance
                - GS, V, JPM
 
    2. Clean Data
        - Remove AH (easy) 
            - Compute returns before removing
            - Make assumption that all valid trading days begins at 9:30-4:00 
                - Data will be kept from 9 - 4
        - Ignore gaps (they won't play a role in calculation anyways)
        - Drop nan values
        - split data in to train & test = 50/50.
            - this is because we only predict 1 timestep in the future and factor the data into p after that time step passes

    3. Control Model
        - Bear/Bull matrix
            - Bulls >= 0
            - Bears < 0
        - Compute Probability on test
        - Compute new probs over all time periods

    4. Other Models
        1. Thresholding
        2. P(Dt, Ht)
    
    5. "Train"
        - Compute initial Probability (whether it is constant or not)
        - Compute initial means and variances

    6. "Test"
        - For each state (in matrix like P), track mean returns and std returns. 
        - For every month in test, simulate n runs, with the predicted return being a normal distribution of the selected state (due to MC).
        - Add new model to Training set

    7. Statistics
        - Strategy - Get a distribution of results at the end of each month, create a Page rank of each stock, create a portfolio

        - Plots
            Like Paper
                - Sections [train, test1, test2 ...]
                - train -> only reals
                - testn -> real and our predictions
                - Can split with red line.

            Quantile of simulations (sample in validate_sim.ipynb)
                - Max & min & median & Q1 & Q3
                - Plot w.r.t real for simulations
            

            Portfolio plots.
                - PNL over time
                - Sharp over time
            
            Others?
            


    tkomeiji20

Future work

Page rank could use different computations, such as correlation