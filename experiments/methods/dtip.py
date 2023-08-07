import pandas as pd
import numpy as np
from gurobipy import Model, quicksum, GRB, Env
from sklearn.metrics import r2_score
from methods.misc.util import load_data_cont_bincat
import math
import time
import traceback


# MIP model from 
# Verwer, Sicco, and Yingqian Zhang. "Learning decision trees with flexible constraints and objectives using integer optimization." CPAIOR 2017.

# Implementation by Jacobus G. M van der Linden

# Bin for scaling 
class Bin:
    
    def __init__(self, left, right, count, label):
        self.left = left
        self.right = right
        self.count = count
        self.label = label
        
    def __repr__(self):
        return "[({},{}), {}, {}]".format(self.left, self.right, self.count, self.label)

# DTIP scaler
class DTIPScaler:
    
    def __init__(self):
        self.bins = []
    
    def fit(self, X, _y):
        for c_ix, c in enumerate(X.columns):
            y = _y.copy()
            x = X[c]
            x = x.drop_duplicates().sort_values()
            y = y[x.index]
            
            removes = []
            for i, ix in enumerate(x.index):
                if i == 0: 
                    prev = ix
                    continue
                if y[ix] == y[prev]:
                    removes.append(ix)
            x = x.drop(index=removes)
            #y = y.drop(index=removes)
            
            if len(x) == 1:
                bins = [(-np.inf, np.inf)]
            else:
                bins = [(-np.inf, (x.iloc[0] + x.iloc[1]) / 2)] \
                    + [((x.iloc[i] + x.iloc[i+1]) / 2, (x.iloc[i+1] + x.iloc[i+2]) / 2) for i in range(len(x) - 2)] \
                    + [((x.iloc[-2] + x.iloc[-1]) / 2, np.inf)] 
                
            _x = X[c]
            counts = [sum((_x >= b[0]) & (_x < b[1])) for b in bins]
            count_sort = np.argsort(counts)[::-1]
            labels = np.zeros((len(count_sort,)))
            last_label = 0
            for i, ix in enumerate(count_sort):
                labels[ix] = last_label
                if last_label <= 0:
                    last_label = last_label * -1 + 1
                else:
                    last_label *= -1
            self.bins.append([])
            for bn, count, label in zip(bins, counts, labels):
                self.bins[c_ix].append(Bin(bn[0], bn[1], count, label))

    def _transform(self, c_ix, v):
        bins = self.bins[c_ix]
        for _bin in bins:
            if v >= _bin.left and v < _bin.right:
                return _bin.label
        assert(False)
                
    def transform(self, X):
        output_columns = []
        for c_ix, c in enumerate(X.columns):
            x = X[c]
            new_c = x.apply(lambda v: self._transform(c_ix, v))
            output_columns.append(new_c)
            
        return pd.concat(output_columns, axis=1)


def _run_dtip(
    time_limit,     # Time limit in seconds
    depth,          # maximum depth of the tree. D=0 is a single leaf node. D=1 is one branching node with two leaf nodes. Etc.
    train_data,     # Path to the training data (csv)
    test_data,      # Path to the test data (csv)
    tune = False    # If true, apply hyper tuning (not implemented)
):
    assert(tune == False)
    verbose = True

    _train_X, train_y, train_info = load_data_cont_bincat(train_data)
    _test_X, test_y, test_info = load_data_cont_bincat(test_data)

    start_time = time.time() # Start timer after reading data
    
    
    scaler = DTIPScaler()
    scaler.fit(_train_X, train_y)
    train_X = scaler.transform(_train_X)
    test_X = scaler.transform(_test_X)

    # Create model
    N = train_X.shape[0]
    F = train_X.shape[1]

    total_train_var = train_y.std()**2
    total_test_var = test_y.std()**2

    params = {
        "LogToConsole": True,
        "Threads": 1, # compute with one thread for fair comparison with other non-parallel methods
        "TimeLimit": time_limit,
        "MemLimit": 75
    }
    with Env(params=params) as env, Model('DTIP', env=env) as model:
        datapoints = _train_X.index
        features = list(range(0, F))
        branching_nodes = [i for i in range(1, 2**depth)]
        leaf_nodes = [2**depth + i for i in range(0, 2**depth)]
        depths = [d+1 for d in range(depth)]
        depth_of_node = [int(math.floor(math.log2(n) + 1e-3)) + 1 for n in branching_nodes]

        # True if path to j goes to the right at level h
        _dlr = [[False for j in branching_nodes + leaf_nodes] for h in range(depth+1)]
        for n in branching_nodes + leaf_nodes:
            if n == 1: continue
            t2 = n
            h = int(math.floor(math.log2(n) + 1e-3))
            while t2 > 1:
                t3 = t2 // 2 # get ancestor
                h -= 1
                if t2 % 2 == 1:
                    _dlr[h][n-1] = False
                else:
                    _dlr[h][n-1] = True
                t2 = t3
            
        def dlr(h, j, r):
            if _dlr[h-1][j-1]:
                return dhr[h, r]
            return 1 - dhr[h, r]

        LF = train_X.min().min()
        UF = train_X.max().max()

        M = train_X.max(axis=1) - LF
        M_ = UF - train_X.min(axis=1) 
        
        LT = train_y.min()
        UT = train_y.max()

        Mt = UT - train_y.min()
        Mt_ = train_y.max() - LT

        # Datapoint r goes  right/left at depth h
        dhr = model.addVars(depths, datapoints, vtype=GRB.BINARY, name="dhr")
        # Feature i is used in node j
        f = model.addVars(features, branching_nodes, vtype=GRB.BINARY, name='f')
        # Threshold at node j
        c = model.addVars(branching_nodes, vtype=GRB.INTEGER, lb=LF, ub=UF, name="c")

        # Prediction at leaf node l
        p = model.addVars(leaf_nodes, vtype=GRB.CONTINUOUS, lb=-np.inf, name='p')

        # Error of row r
        e = model.addVars(datapoints, vtype=GRB.CONTINUOUS, name='e')


        # One feature used in every branching node
        model.addConstrs(quicksum([f[i,j] for i in features]) == 1 for j in branching_nodes)

        # Does row r take the left or right branch?    
        model.addConstrs(quicksum([M[r] * dlr(h, j, r) for h in range(1, depth_of_node[j-1])]) + M[r]*dhr[depth_of_node[j-1], r] \
            + quicksum([train_X.loc[r, train_X.columns[i]] * f[i, j] for i in features]) <= M[r] * (depth_of_node[j-1]) + c[j] - 1 for r in datapoints for j in branching_nodes)
            
        model.addConstrs(quicksum([M_[r] * dlr(h, j, r) for h in range(1, depth_of_node[j-1])]) - M_[r]*dhr[depth_of_node[j-1], r] \
            - quicksum([train_X.loc[r, train_X.columns[i]] * f[i, j] for i in features]) <= M_[r] * (depth_of_node[j-1] - 1) - c[j] for r in datapoints  for j in branching_nodes)

        # Compute the error
        model.addConstrs(quicksum([Mt * dlr(h, j, r) for h in range(1, depth+1)])  + p[j] - train_y[r] <= e[r] + Mt * depth for r in datapoints for j in leaf_nodes)
        model.addConstrs(quicksum([Mt_ * dlr(h, j, r) for h in range(1, depth+1)]) - p[j] + train_y[r] <= e[r] + Mt_ * depth for r in datapoints  for j in leaf_nodes)


        # minimize Sum Absolute Error
        model.setObjective(quicksum(e), GRB.MINIMIZE)
        model.optimize()
        
        duration = time.time() - start_time

        if model.Status == GRB.TIME_LIMIT:
            gap = model.MIPGap
            if verbose:
                print("Time out")
                print("MIP gap: ", gap * 100)
            return {"time": time_limit + 1, "train_r2": -1, "test_r2": -1, "leaves": -1, "terminal_calls": -1}

        def get_yhat(_X):
            yhat = []
            for i in _X.index:
                t = 1
                while not t in leaf_nodes:
                    feature = np.argmax([f[i, t].X for i in features]) 
                    if _X.loc[i, _X.columns[feature]] < c[t].X:
                        t *= 2
                    else: 
                        t = t * 2 + 1
                yhat.append(p[t].X)
            yhat = pd.Series(yhat, index=_X.index)
            return yhat

        train_yhat = get_yhat(train_X)
        train_r2 = r2_score(train_y, train_yhat)
        if verbose:
            print("\nTrain MSE: ", -(train_r2 - 1) * total_train_var)
            print("Train R^2: ", train_r2)

        test_yhat = get_yhat(test_X)
        test_r2 = r2_score(test_y, test_yhat)
        if verbose:
            print("\nTest MSE: ", -(test_r2 - 1) * total_test_var)
            print("Test R^2: ", test_r2)
    
    return {"time": duration, "train_r2": train_r2, "test_r2": test_r2, "leaves": 2**depth, "terminal_calls": -1}

    
def run_dtip(*args, **kwargs):
    try:
        return _run_dtip(*args, **kwargs)
    except Exception:
        traceback.print_exc()
        return {"time": -1, "train_r2": -1, "test_r2": -1, "leaves": -1, "terminal_calls": -1}
