import pandas as pd
# get the path to the bag file
from pathlib import Path
import os

def get_bag_path(row, base_path):
    if 'SFW' in row['method']:
        bag_prefix = 'DumbNav'
        if 'SAC' in row['method']:
            bag_prefix = 'SFW/SAC'
    else:
        bag_prefix = 'SFW/SAC'
    model_path = base_path + '/' + bag_prefix + '/' + row['method'] + '/' + row['experiment']
    model_path = Path(model_path)
    bag_path = os.path.join(model_path, os.listdir(model_path)[-1], 'bags', f'episode_{row["episode"]}',f'episode_{row["episode"]}_0.db3')
    return bag_path
