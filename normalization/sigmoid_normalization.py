import pandas as pd
import numpy as np
import math

vkp = pd.read_excel('VeriKumesi.xlsx')
svk = np.array(vkp)

j = 0
while j < 3:
  listMax = float(max(vk[:, j])) # type: ignore
  svk[:, j] = [i/listMax for i in vk[:, j]] # type: ignore
  svk[:, j] = [1/(1+math.exp(-i)) for i in vk[:, j]] # type: ignore
  j += 1

df = pd.DataFrame (svk)
df.to_excel('sigmoid.xlsx',index=False)