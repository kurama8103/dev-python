# %%
#!/usr/bin/python
# -*- coding: utf-8 -*-
import datetime as dt
import subprocess
import sys
import warnings

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pylab import rcParams
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.precision', 2)
pd.options.display.float_format = "{:,.4f}".format
np.set_printoptions(suppress=True)  # 指数表記
np.set_printoptions(precision=2)


rcParams['figure.figsize'] = 12, 3
plt.figure()
plt.style.use('seaborn')
japanize_matplotlib.japanize()

sns.set_style('whitegrid')


def main(arg: str) -> str:
    return arg


if __name__ == '__main__':
    if 'ipykernel_launcher.py' in sys.argv[0]:
        arg1 = 'arg'
    else:
        arg1 = sys.argv[1]
    main(arg1)
