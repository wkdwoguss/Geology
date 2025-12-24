import pandas as pd
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from operate import merging, graphing, showMLP

oceanpath = r'Western_Australia\PERTH_AIRPORT,_AS.csv'
landpath = r'Western_Australia\KALGOORLIE_BOULDER_AIRPORT,_AS.csv'

ocean = pd.read_csv(oceanpath)
land = pd.read_csv(landpath)

merged = merging(ocean, land)
if not os.path.isfile(r'Western_Australia\Result.csv'): merged.to_csv(r'Western_Australia\Result.csv', index=False)

graphing(merged, "Western Australia")

showMLP(merged, r'Western_Australia\weights_Western_Australia.pth')