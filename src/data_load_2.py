
import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/new_final/df_20.csv')
# df['person_id'] = (
#     df['person_id'].astype(str)
#     + '_'
#     + df['giro'].astype(str)
# )

df_gt = pd.read_csv('/content/drive/MyDrive/new_final/corr_df_gt_20.csv')
# df_gt['person_id'] = (
#     df_gt['person_id'].astype(str)
#     + '_'
#     + df_gt['giro'].astype(str)
# )
df_gt = df_gt[df_gt['dur'] >= 10]  # discard segments with duration < 10s
exhibits_df = pd.read_csv('/content/drive/MyDrive/new_final/POIs.csv')
# df.to_csv("df_20.csv")
# df_gt.to_csv("df_gt_20.csv")
# exhibits_df.to_csv("POIs.csv")