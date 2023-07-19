import re
import subprocess
import sys

import pandas as pd
import numpy as np
from gurobipy import Model, quicksum, GRB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import time

# MIP model from 
# Bertsimas, Dimitris, and Jack Dunn. "Optimal classification trees." Machine Learning 106 (2017): 1039-1082.
# Dunn, Jack William. Optimal trees for prediction and prescription. Diss. Massachusetts Institute of Technology, 2018.
# Implementation by Jacobus G. M van der Linden
def run_ort(
    time_limit,     # Time limit in seconds
    depth,          # maximum depth of the tree. D=0 is a single leaf node. D=1 is one branching node with two leaf nodes. Etc.
    train_data,     # Path to the training data (csv)
    test_data,      # Path to the test data (csv)
    cp,             # Cost complexity parameter
    linear_model,   # True: train piecewise linear model, False: train piecewise constant model
    lasso_penalty = 0,  # The lasso penalization factor in the Elastic Net (only for linear model). Default = 0
    metric = "MAE", # Which metric to optimize, either MSE or MAE. MAE is default. The evaluation score that is returned is MSE and R^2
    tune = False    # If true, apply hyper tuning (not implemented)
):
    assert(tune == False)
    verbose = True

    csv_sep = " "      # Column seperator in files.   Change if needed
    csv_header = None   # Header paramater for pandas. Change if needed
    label_column = "0"  # Name of the label column.    Change if needed

    train_df = pd.read_csv(train_data, sep=csv_sep, header=csv_header)
    test_df = pd.read_csv(test_data, sep=csv_sep, header=csv_header)

    start_time = time.time() # Start timer after reading data

    if train_df.columns.dtype == np.int64:
        label_column = int(label_column)

    # Get X and y     
    _train_X = train_df.drop(columns=[label_column])
    train_y = train_df[label_column]
    _test_X = test_df.drop(columns=[label_column])
    test_y = test_df[label_column]
    
    # Normalize X
    scaler = MinMaxScaler()
    scaler.fit(_train_X)
    train_X = scaler.transform(_train_X)
    test_X = scaler.transform(_test_X)

    
    # Create model
    N = train_X.shape[0]
    F = train_X.shape[1]

    # Require at least 10 instances in a leaf node for every independent variable
    min_support = F * 10 

    total_train_var = train_y.std()**2
    total_test_var = test_y.std()**2

    model = Model('ORT')
    model.params.LogToConsole = True
    model.params.Threads = 1 # compute with one thread for fair comparison with other non-parallel methods
    model.params.TimeLimit = time_limit

    datapoints = _train_X.index
    features = list(range(0, F))
    branching_nodes = [i for i in range(1, 2**depth)]
    leaf_nodes = [2**depth + i for i in range(0, 2**depth)]

    z = model.addVars(datapoints, leaf_nodes, vtype=GRB.BINARY, name='z')
    d = model.addVars(branching_nodes, vtype=GRB.BINARY, lb=1, name='d')
    a = model.addVars(branching_nodes, features, vtype=GRB.BINARY, name='a')
    if linear_model:
        r = model.addVars(leaf_nodes, features, vtype=GRB.BINARY, name='r')
        l = model.addVars(leaf_nodes, vtype=GRB.BINARY, name='l')

    if linear_model:
        
        if metric == "MAE":
            M = 2 * (total_train_var + max(abs(train_y)))
        else:
            M = 2 * (math.sqrt(total_train_var) + max(abs(train_y)))
        f = model.addVars(datapoints, vtype=GRB.CONTINUOUS, lb = -M, name='f')
        
    else:
        f = model.addVars(datapoints, vtype=GRB.CONTINUOUS, lb=min(train_y), ub=max(train_y), name='f')
        M = (max(train_y) - min(train_y))
        
    if metric == "MAE" and not linear_model:
        L = model.addVars(datapoints, vtype=GRB.CONTINUOUS, lb=0, ub=M, name='L')    
    else:
        L = model.addVars(datapoints, vtype=GRB.CONTINUOUS, lb=0, name='L')
    C = model.addVar(vtype=GRB.INTEGER, lb=0, ub=2**depth-1, name='C')
    b = model.addVars(branching_nodes, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='b')
    if linear_model:
        beta = model.addVars(leaf_nodes, features, vtype=GRB.CONTINUOUS, lb=-np.inf, name='beta')
        b0 = model.addVars(leaf_nodes, vtype=GRB.CONTINUOUS, lb=-np.inf, name='b0')
    else:
        b0 = model.addVars(leaf_nodes, vtype=GRB.CONTINUOUS, lb=min(train_y), ub=max(train_y), name='b0')


    if metric == "MAE":
        # MAE Loss per instance
        model.addConstrs(L[i] >= f[i] - train_y[i] for i in datapoints)
        model.addConstrs(L[i] >= -f[i] + train_y[i] for i in datapoints)
    else:
        # MSE Loss per instance
        model.addConstrs(L[i] >= (f[i] - train_y[i]) * (f[i] - train_y[i]) for i in datapoints)

    # Leaf decision
    if linear_model:
        model.addConstrs(f[i] - b0[t] - quicksum([beta[t, p]*train_X[i, p] for p in features]) >= -M*(1-z[i, t]) for i in datapoints for t in leaf_nodes)
        model.addConstrs(f[i] - b0[t] - quicksum([beta[t, p]*train_X[i, p] for p in features]) <=  M*(1-z[i, t]) for i in datapoints for t in leaf_nodes)
        
        Mr = 100
        model.addConstrs(-Mr * r[t, p] <= beta[t, p] for p in features for t in leaf_nodes)
        model.addConstrs( Mr * r[t, p] >= beta[t, p] for p in features for t in leaf_nodes)
        
    else:
        model.addConstrs(f[i] - b0[t] >= -M*(1-z[i, t]) for i in datapoints for t in leaf_nodes)
        model.addConstrs(f[i] - b0[t] <=  M*(1-z[i, t]) for i in datapoints for t in leaf_nodes)

    # Branching decision
    eps = [] # smallest diff for each feature
    for p in features:
        uniqs = sorted(np.unique(train_X[:, p]))
        if len(uniqs) == 1:
            model.addConstr(quicksum([a[i, p] for i in branching_nodes]) == 0)
            eps.append(np.nan)
        else:
            diffs = [uniqs[i] - uniqs[i-1] for i in range(1, len(uniqs))]
            eps.append(min(diffs))
    eps_max = max(eps)
    for t in leaf_nodes:
        left_ancestors = []
        right_ancestors = []
        t2 = t
        while t2 > 1:
            t3 = t2 // 2 # get ancestor
            if t2 % 2 == 1:
                right_ancestors.append(t3)
            else:
                left_ancestors.append(t3)
            t2 = t3
        if len(left_ancestors) > 0:
            model.addConstrs(quicksum([(train_X[i, p]+eps[p])*a[m, p] for p in features]) <= b[m] + (1 + eps_max) * (1-z[i,t])
                        for i in datapoints for m in left_ancestors)
        if len(right_ancestors) > 0:
            model.addConstrs(quicksum([train_X[i, p]*a[m, p]          for p in features]) >= b[m] - (1-z[i,t])
                        for i in datapoints for m in right_ancestors)

    model.addConstrs(quicksum([a[i, p] for p in features]) == d[i] for i in branching_nodes)
    model.addConstrs(b[i] <= d[i] for i in branching_nodes)
    model.addConstrs(d[t] <= d[t//2] for t in branching_nodes[1:])

    # One leaf node per instance
    model.addConstrs(quicksum([z[i,t] for t in leaf_nodes]) == 1 for i in datapoints)

    model.addConstr(C == quicksum(d))

    if linear_model:
        # minimum leaf node size constraint
        model.addConstrs(z[i, t] <= l[t] for t in leaf_nodes for i in datapoints)
        model.addConstrs(quicksum([z[i,t] for i in datapoints]) >= min_support * l[t] for t in leaf_nodes)

    # minimize Sum Absolute Error
    if linear_model:
        model.setObjective((quicksum(L) / total_train_var + cp * C + lasso_penalty*quicksum([r[t, p] for t in leaf_nodes for p in features])), GRB.MINIMIZE)
    else:
        model.setObjective((quicksum(L) / total_train_var + cp * C ), GRB.MINIMIZE)

    try:
        model.optimize()
    except Exception as exp:
        print(exp.stdout.decode(), file=sys.stderr, flush=True)
        return {"time": -1, "train_mse": -1, "test_mse": -1, "leaves": -1, "terminal_calls": -1}

    duration = time.time() - start_time

    if model.Status == GRB.TIME_LIMIT:
        gap = model.MIPGap
        if verbose:
            print("Time out")
            print("MIP gap: ", gap * 100)
        return {"time": time_limit + 1, "train_mse": -1, "test_mse": -1, "leaves": -1, "terminal_calls": -1}

    try:
        yhat = pd.Series(model.getAttr("X", f))
        zs = model.getAttr("X", z)
        zs = pd.DataFrame({"ix": i, "leaf": j, 'value': z[i,j].X} for (i,j) in zs)
        if verbose:
            for t in leaf_nodes:
                print("Leaf ", t, ": ", int(sum(zs[zs["leaf"] == t]["value"])), " instances")

        n_branching_nodes = sum([d[i].X for i in branching_nodes])

        train_mse = mean_squared_error(train_y, yhat)
        if verbose:
            print("\nTrain MSE: ", train_mse)
            print("Train R^2: ", 1 - train_mse / total_train_var)

        ytest_hat = []
        for i in _test_X.index:
            t = 1
            while not t in leaf_nodes:
                if sum([a[t, p].X * test_X[i, p] for p in features]) <= b[t].X:
                    t *= 2
                else:
                    t = t * 2 + 1
            if linear_model:
                ytest_hat.append(b0[t].X + sum([beta[t, p].X * test_X[i, p] for p in features]))
            else:
                ytest_hat.append(b0[t].X)
        ytest_hat = pd.Series(ytest_hat, index=_test_X.index)

        test_mse = mean_squared_error(test_y, ytest_hat)
        if verbose:
            print("\nTest MSE: ", test_mse)
            print("Test R^2: ", 1 - test_mse / total_test_var)

        return {"time": duration, "train_mse": train_mse, "test_mse": test_mse, "leaves": int(n_branching_nodes + 1), "terminal_calls": -1}
    
    except Exception as exp:
        print(exp.stdout.decode(), file=sys.stderr, flush=True)
        return {"time": -1, "train_mse": -1, "test_mse": -1, "leaves": -1, "terminal_calls": -1}

    
