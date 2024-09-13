# %%
import numpy as np
import pandas as pd
import pickle
from pathlib import Path


# %%

# pickle_file = Path("/home/andrea/workspaces/appl_ws/metrics.pickle")
pickle_file = Path("/home/andrea/workspaces/appl_ws/output.pickle")
with open(pickle_file, "rb") as f:
    data = pickle.load(f)
exp = list(data.keys())
# df_exp = pd.DataFrame()
print(exp)
# print(data[exp[0]])
print(data[exp[0]].keys())
print(data[exp[0]]["goal"])
print(data[exp[0]]["obstacles"])
df_odometry = pd.DataFrame(data[exp[0]]["odometry"])
df_people = pd.DataFrame(data[exp[0]]["people"])
df_agents_list = data[exp[0]]["agents"]
df_agent_0 = pd.DataFrame(df_agents_list[0])
df_obstacles = data[exp[0]]["obstacles"]
transforms = data[exp[0]]["transforms"]
goal = data[exp[0]]["goal"]

# df_exp = pd.DataFrame(data[exp[0]])
# df_exp = pd.Series(data[exp[0]][0])
# df_exp = pd.Series()

# for e in exp:
#     # print(e)
#     # print(data[e])
#     # print(data[e][0].keys())
#     df_exp_ = pd.Series(data[e][0])
#     df_exp = pd.concat([df_exp, df_exp_], axis=1)

# df_exp = df_exp.T  # Transpose the dataframe
# df_exp = df_exp.dropna()
# df_exp = df_exp.reset_index(drop=True)


# %%
pickle_file = Path("/home/andrea/workspaces/appl_ws/metrics.pickle")
with open(pickle_file, "rb") as f:
    data = pickle.load(f)
