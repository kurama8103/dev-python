# %%
import subprocess
import sys
import quantstats as qs
import pandas as pd


def quantstats_html(filepath_csv, bm=None):
    qs.extend_pandas()
    df = pd.read_csv(filepath_csv, index_col=0,
                     parse_dates=True, usecols=[0, 1]).iloc[:, 0]
    if min(df) <= 0:
        pass
    else:
        df = df.pct_change().dropna()
    filename_html = df.name+'-' + \
        df.index[0].strftime('%Y%m%d')+'-' + \
        df.index[-1].strftime('%Y%m%d')+'.html'
    qs.reports.html(df, bm=bm, rf=0., title=df.name,
                    download_filename=filename_html, output='.')
    return filename_html


if __name__ == '__main__':
    if 'ipykernel_launcher.py' in sys.argv[0]:
        filepath_csv = '../data/test.csv'
    else:
        filepath_csv = sys.argv[1]
    filename_html = quantstats_html(filepath_csv)
    subprocess.call(['open', filename_html])

# %%
