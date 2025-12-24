import pandas as pd
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from operate import merging, graphing, showMLP

oceanpath = r'Eastern_Australia\SYDNEY_AIRPORT_AMO,_AS.csv'
landpath = r'Eastern_Australia\DUBBO_AIRPORT_AWS,_AS.csv'

ocean = pd.read_csv(oceanpath)
land = pd.read_csv(landpath)

merged = merging(ocean, land)
if not os.path.isfile(r'Eastern_Australia\Result.csv'): merged.to_csv(r'Eastern_Australia\Result.csv', index=False)

graphing(merged, "Eastern_Australia")

showMLP(merged, r'Eastern_Australia\weights_Eastern_Australia.pth')