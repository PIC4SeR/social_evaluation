# %%
import numpy as np
import pandas as pd
import pickle
from pathlib import Path


# %%

pickle_file = Path("/home/andrea/workspaces/appl_ws/metrics.pickle")
with open(pickle_file, "rb") as f:
    data = pickle.load(f)
exp = list(data.keys())
# df_exp = pd.DataFrame()
print(exp)
print(data[exp[0]])
df_exp = pd.DataFrame(data[exp[0]])
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
