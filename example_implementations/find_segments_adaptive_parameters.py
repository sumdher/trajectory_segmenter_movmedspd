
import time
import pandas as pd

import sys
sys.path.insert(1, 'C:\\ZZZZZ Pette ZZZZZ\\TrajectorySegmenter')

from src.segmenter import MovMedSpeedSegmenter
from src.evaluater import MatchingRetrieverNew

df = pd.read_csv(r'data\df_20.csv')
df_gt = pd.read_csv(r'data\df_gt_20.csv')
df_gt = df_gt[df_gt['dur'] >= 10]  # discard segments with duration < 10s
exhibits_df = pd.read_csv(r'data\POIs.csv')

alpha = 'auto'  # Finds window where entropy difference doesn't change much.
window_size = 'auto'  # Must be set to 'auto' if alpha is 'auto'. Unique windows.
steep = 0.15  # Scipy's find_peaks height threshold. 
              # Line intersections steep % decrease of (1 - .steep)%.
tol = 2.00  # Extend terminal points until speed >= tol*std.
cutoff_duration = 10  # Discard segments < cutoff_duration
mode = "median"  # Moving window function. Can also be: 'mean' or 'ewm'/'ema'.
path = "/content/final_segments.csv"  # Output file path.


start_time = time.time()

seg_med = MovMedSpeedSegmenter(df, exhibits_df)
seg_med.process_trajectory_data()
seg_med.find_elbows(min=1, max=151, step=3, plot=False)

# alpha = seg_med.elbow_points_df['second_elbow'].mean()  

seg_med.optimal_window_filter(alpha='auto')
seg_med.segment_graph(mode=mode, window_size="auto", steep=steep)
seg_med.closest_POI_from_geom()
before_refining_seg = seg_med.merge_segments_df(cutoff_duration)
result_segments = seg_med.refine_end_segment_boundaries(tol, before_refining_seg)
# seg_med.export_segments_csv(path=path)

end_time = time.time()
print(f'Execution time: {(end_time - start_time)}')

# Evaluation

"""
Uncomment the code below if ground truth is present.
The below code will evaluate the segments detected above with the real ones.
Various statistical measures can be seen.
"""

# eval_med = MatchingRetrieverNew(result_segments, df_gt)
# eval_med.process_prep()  # Processes and sets the data up.
# eval_results = eval_med.find_match(alfa=0.5)

# if eval_results is not None:
#     print(f"Ends refined with {tol}*std: ",
#           eval_results['avg_jacc'].mean(),
#           "+-",
#           eval_results['avg_jacc'].std()
#     )

#     print(eval_results[['person_id', 'f_measure', 'avg_jacc']])
#     print(eval_results[['f_measure', 'avg_jacc']].mean())
#     print(eval_results[['recall', 'precision', 'accuracy']].mean())

#     print(
#         eval_results[
#             ["true_positive", "true_negative",
#              "false_positive", "false_negative"]
#         ].sum()
#     )
