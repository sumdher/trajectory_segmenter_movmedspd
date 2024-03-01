"""
This only works as intended in Python Notebooks.
Originally, developped in Google Colab.
"""

import pandas as pd

from src.segmenter import MovMedSpeedSegmenter

df = pd.read_csv('data\df_20.csv')
df_gt = pd.read_csv('data\df_gt_20.csv')

df_gt = df_gt[df_gt['dur'] >= 10]  # discard segments with duration < 10s
exhibits_df = pd.read_csv('data\POIs.csv')

tol = 2.00

# Pass the ground truth dataframe ("df_gt") in the constructor
# to see the real segments in the graph
object2 = MovMedSpeedSegmenter(df, exhibits_df)
object2.process_trajectory_data()

# Pass real=False if you don't want to visualize the real stops.
# Pass the tolerance (tol) for "refine_end_segment_boundaries"
object2.scipy_peaks_plot(real=False, tol=tol)
