# %%
#!/usr/bin/python
# -*- coding: utf-8 -*-
import datetime as dt
import subprocess
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pylab import rcParams

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.precision', 2)
pd.options.display.float_format = "{:,.4f}".format

plt.style.use('seaborn')
rcParams['figure.figsize'] = 15, 3


def main(arg: str) -> str:
    return arg


if __name__ == '__main__':
    if 'ipykernel_launcher.py' in sys.argv[0]:
        arg = 'arg'
    else:
        arg = sys.argv[1]
    main(arg)
