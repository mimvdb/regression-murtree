import pandas as pd
from scipy.stats import gmean

df = pd.read_csv("results/scale_results/iai-vs-srt-results.csv")

g = df.groupby(["train_data", "method", "depth"])["time"].mean().unstack("method")

print ("IAI vs SRT-C")
print( gmean(g["iai"] / g["streed_pwc_kmeans1_tasklb1_lb1_terminal1"]))

print( gmean( (g["iai"] / g["streed_pwc_kmeans1_tasklb1_lb1_terminal1"]).unstack("depth") , axis=0, nan_policy="omit") )

print ("IAI vs SRT-L")
print( gmean(g["iai_l"] / g["streed_pwl"], nan_policy="omit"))

print( gmean( (g["iai_l"] / g["streed_pwl"]).unstack("depth"), axis=0, nan_policy="omit"))