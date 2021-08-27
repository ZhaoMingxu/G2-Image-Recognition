import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

files = glob.glob('E:\G2\可视化数据' + '\*.csv')
steps = []
values = []
fig = plt.figure()
ax = fig.add_subplot(111)
for i, file in enumerate(files):
        df = pd.read_csv(file)
        step = df['Step'].values
        wall = df['Wall time'].values
        value = df['Value'].values
        steps.append(step)
        values.append(value)
        ax.plot(wall, value)

plt.show()

