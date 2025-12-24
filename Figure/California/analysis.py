import pandas as pd
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from operate import merging, graphing, showMLP

oceanpath = r'California\SAN_FRANCISCO_INTERNATIONAL_AIRPORT,_CA_US.csv'
landpath = r'California\SACRAMENTO_METROPOLITAN_AIRPORT,_CA_US.csv'

ocean = pd.read_csv(oceanpath)
land = pd.read_csv(landpath)

merged = merging(ocean, land)
if not os.path.isfile(r'California\Result.csv'): merged.to_csv(r'California\Result.csv', index=False)

graphing(merged, "California")

showMLP(merged, r'California\weights_California.pth')