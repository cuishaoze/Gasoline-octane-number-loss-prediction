import pandas as pd
import numpy as np

# data import
df = pd.read_csv('/Data.csv')

# normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
StandData = scaler.fit_transform(df.iloc[:,0:-1])
StandData = pd.DataFrame(StandData)
StandData = pd.concat([StandData, df['RON_loss']],axis=1)
StandData.columns = df.columns

# DE-based optimization
from DE import DifferentialEvolutionAlgorithm
import warnings
warnings.filterwarnings('ignore')
# DE-Stacking
bound = np.array([[1,1,1,0.1,1],[250,10,250,1.0,250]])
de = DifferentialEvolutionAlgorithm(50, 5, bound, 50, [0.8, 0.6], StandData)
de.solve()
