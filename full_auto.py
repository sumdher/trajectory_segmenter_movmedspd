
from classes import matchingretr, movmedspd
import time
import pandas as pd

df = pd.read_csv('data\df_20.csv')
df_gt = pd.read_csv('data\df_gt_20.csv')
df_gt = df_gt[df_gt['dur'] >= 10]  # discard segments with duration < 10s
exhibits_df = pd.read_csv('data\POIs.csv')


alpha = 'auto'
window_size = 'auto'  # Must be set to 'auto' if alpha is 'auto'
tol = 2.00 
cutoff_duration = 10
mode = "median"  # Moving window function. Can also be: 'mean' or 'ewm'/'ema'.
path = "/content/final_segments.csv"  # Output file path


start_time = time.time()

seg_med = movmedspd.MovMedSpeedSegmenter(df, exhibits_df)
seg_med.process_trajectory_data()
seg_med.find_elbows(min=1, max=151, step=3, plot=False)

# alpha = seg_med.elbow_points_df['second_elbow'].mean()  # More in "WindowAnalysis.py"

seg_med.optimal_window_filter(alpha='auto')
seg_med.segment_graph(mode="median", window_size="auto")
seg_med.closest_exhibit_from_geom()  # finds closest POI
before_refining_seg_df = seg_med.club_segments_df(cutoff_duration)
result_segments = seg_med.refine_end_segment_boundaries(tol, before_refining_seg_df)
# seg_med.export_segments_csv(path=path)

end_time = time.time()

print(f'Execution time: {(end_time - start_time)}')


eval_med = matchingretr.MatchingRetrieverNew(result_segments, df_gt)
eval_med.process_prep()  # Processes and sets the data up.
eval_results = eval_med.find_match(alfa=0.5)  

if eval_results is not None:
    print(f"Ends refined with {tol}*std: ",
          eval_results['avg_jacc'].mean(),
          "+-",
          eval_results['avg_jacc'].std()
    )

    print(eval_results[['person_id', 'f_measure', 'avg_jacc']])
    print(eval_results[['f_measure', 'avg_jacc']].mean())
    # print(eval_results[['recall', 'precision', 'accuracy']].mean())
    
    print(
        eval_results[
            ["true_positive", "true_negative",
             "false_positive", "false_negative"]
        ].sum()
    )