
import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/ground_truth_09/tj_all.csv')

df_gt = pd.read_csv('/content/drive/MyDrive/ground_truth_09/gt_stops.csv')

exhibits_df = pd.read_csv('/content/drive/MyDrive/ground_truth_09/POIs.csv')

df.rename(
    columns={
        'time_string': 'time_stamp'
        }, inplace=True
)

df_gt.rename(
    columns={
        'centroid_x_kf': 'x_centroid',
        'centroid_y_kf': 'y_centroid',
        'exhibit': 'gt_poi'
    }, inplace=True
)

exhibits_df['x'] = exhibits_df['x'] / 100
exhibits_df['y'] = exhibits_df['y'] / 100
exhibits_df = exhibits_df.sort_values(by=['name'])