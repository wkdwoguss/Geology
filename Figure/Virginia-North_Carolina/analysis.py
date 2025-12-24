import pandas as pd
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from operate import merging, graphing, showMLP

oceanpath = r'Virginia-North_Carolina\NORFOLK_INTERNATIONAL_AIRPORT,_VA_US.csv'
landpath = r'Virginia-North_Carolina\RICHMOND_INTERNATIONAL_AIRPORT,_VA_US.csv'

ocean = pd.read_csv(oceanpath)
land = pd.read_csv(landpath)

merged = merging(ocean, land)
if not os.path.isfile(r'Virginia-North_Carolina\Result.csv'): merged.to_csv(r'Virginia-North_Carolina\Result.csv', index=False)

graphing(merged, "Virginia - North Carolina")

showMLP(merged, r'Virginia-North_Carolina\weights_Virginia-North_Carolina.pth')