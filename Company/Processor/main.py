import pandas as pd
import numpy as np
import scipy as sc
import Regress as nsr
import matplotlib.pylab as plt
from datetime import datetime
fund_data = pd.read_excel('SmoothnessData.xlsx','Sheet1')
assets_data = pd.read_excel('SmoothnessData.xlsx','Sheet2')

funds = fund_data.iloc[2:62,0:158].values
assets = assets_data.iloc[2:62,10:12].values
dates = assets_data.iloc[2:62].index.date
n = np.size(assets,1)
T = np.size(assets,0)
assets = np.mat(assets)
funds = np.mat(funds)
fund = funds[:,0]

lam = 1000
# set a fund to be analyzed
fund = funds[:,2]

# Cross-validation (Leave-one_out procedure)
assets = np.mat(assets)
funds = np.mat(funds)
arrR2, r2, fund_out = nsr.LeaveOneOut(assets,fund,lam)
cumfund = nsr.CumReturn(fund)
cumfund_out = nsr.CumReturn(fund_out)
