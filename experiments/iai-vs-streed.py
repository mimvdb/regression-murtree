import pandas as pd

df = pd.read_csv("results/results-streed-vs-iai.csv")

g = df.groupby(["train_data", "method", "depth"])["time"].mean().unstack("method")

print( (g["iai"] / g["streed_pwc_kmeans1_tasklb1_lb1_terminal1"]).mean())

print( (g["iai"] / g["streed_pwc_kmeans1_tasklb1_lb1_terminal1"]).unstack("depth").mean())

print( (g["iai_l"] / g["streed_pwl"]).mean(skipna=True))

print( (g["iai_l"] / g["streed_pwl"]).unstack("depth").mean(skipna=True))