import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

abspath = os.path.dirname(os.path.abspath(__file__))
map_name = 'test_map_l1'
excel_path = abspath + './{}.xlsx'.format(map_name)
raw_data = pd.read_excel(r'{}'.format(excel_path), header=None)
map = np.array(raw_data.values, dtype=np.uint8)
print("map_size={}".format(map.shape))
np.save('./{}.npy'.format(map_name), map)

plt.figure(figsize=(6, 6))
sns.heatmap(map, cmap='Greys', cbar=False)
plt.show()
