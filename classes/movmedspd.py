
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
from ipywidgets import VBox, Layout, Button, Output, IntText
import ipywidgets as widgets
from IPython.display import display, clear_output
from kneed import KneeLocator
from pulp import *
from scipy.spatial.distance import cdist, jensenshannon
from scipy.stats import entropy
from shapely.geometry import Point
from shapely.wkb import loads


class MovMedSpeedSegmenter:
    """
    Extracts segments from spatio-temporal trajectory data. 
    This process requires points of interest (POI) data, including
    positions, to identify the nearest POI to each segment. 
    It aggregates consecutive segments associated with the same POI
    in proximity.
    
    Additional features include plotting capabilities for 
    visualizing the speed signal, segments, filtered speed signal,
    and the entropy difference elbow curve over a specified range 
    of window lengths for analysis purposes.
    """

    DEFAULT_WINDOW_SIZE = 50
    DEF_DIST = 20  # distance threshold for Scipy find_peaks()
    DEF_WID = 10  # width threshold for Scipy find_peaks()

    def __init__(self, df, exhibits_df, df_gt=None):
        self.logger = self._setup_logging(level=logging.INFO)
        self.traj_df = df.copy()  # input trajectories dataframe
        self.exhibits_df = exhibits_df.copy()  # POIs dataframe
        self.groundtruth_df = (  #  Groundtruth segments dataframe
            df_gt.copy()
            if df_gt is not None
            else None
        )
        self.final_segments_df = None  # Output segments dataframe
        self.segments_df = None  # Output before merging and duration cut
        self.red_peaks_ = None
        self.blue_peaks_ = None
        self.red_peaks = None
        self.red_heights = None
        self.blue_peaks = None
        self.blue_heights = None
        self.window_size = 0
        self.stats_dict = None

    @staticmethod
    def _setup_logging(level=logging.INFO):
        logger = logging.getLogger("MovMedSpeedSegmenter")
        logger.setLevel(level)

        if logger.handlers:
            for handler in logger.handlers:
                handler.setLevel(level)
                formatter = logging.Formatter("%(levelname)s - %(message)s")
                handler.setFormatter(formatter)
        else:
            ch = logging.StreamHandler()
            ch.setLevel(level)
            formatter = logging.Formatter("%(levelname)s - %(message)s")
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        logger.propagate = False
        return logger

    def process_trajectory_data(self) -> pd.DataFrame:
        """
        1. Adds features to traj_df: maonly "speed"
        2. Sorts, converts timestamps to datetime objects.
        """
        self.stats_df = pd.DataFrame(
            columns=['person_id', 'speed_entr', 'den_speed_entr']
        )

        def _calculate_speed(group):
            group["duration"] = (
                group["time_stamp"]
                .diff()
                .fillna(pd.Timedelta(seconds=0))
                .dt.total_seconds()
            )
            group["distance"] = (
                (
                    (group["x_kf"].shift(-1) - group["x_kf"]) ** 2
                    + (group["y_kf"].shift(-1) - group["y_kf"]) ** 2
                )** 0.5
            ).round(2)
            group["speed"] = (group["distance"] / group["duration"]).round(2)
            group["speed"] = group["speed"].replace(
                [float("inf"), -float("inf")], 0
                )
            return group

        self.logger.info(f"Total rows loaded: {len(self.traj_df)}")

        self.traj_df["time_stamp"] = pd.to_datetime(self.traj_df["time_stamp"])
        self.traj_df = self.traj_df.sort_values(by=["person_id", "time_stamp"])
        self.traj_df["time_stamp"] = pd.to_datetime(self.traj_df["time_stamp"])
        self.traj_df["n_point_id"] = (
            self.traj_df.groupby("person_id").cumcount() + 1
            )
        self.traj_df = self.traj_df.groupby(
            "person_id", group_keys=False
            ).apply(_calculate_speed)        
        
        #  Mark real segments from groundtruth_df in traj_df, column "label"
        #  for visualization purposes. Maybe ML?
        if self.groundtruth_df is not None:
            self.traj_df["label_ground"] = 0
            self.groundtruth_df = self.groundtruth_df.sort_values(
                by=["person_id", "start"]
                )
            self.groundtruth_df["start"] = (
                pd.to_datetime(self.groundtruth_df["start"])
                )
            self.groundtruth_df["end"] = pd.to_datetime(
                self.groundtruth_df["end"]
                )
            
            for index_gt, row_gt in self.groundtruth_df.iterrows():
                start_time = pd.to_datetime(row_gt["start"])
                end_time = pd.to_datetime(row_gt["end"])
                matching_rows = self.traj_df[
                    (self.traj_df["time_stamp"] >= start_time)
                    & (self.traj_df["time_stamp"] <= end_time)
                    & (self.traj_df["person_id"] == row_gt["person_id"])
                ]
                self.traj_df.loc[matching_rows.index, "label_ground"] = row_gt[
                    "gt_labeling"
                ]

        return self.traj_df.copy()

    def _make_stats(self, window):
        
        def _calculate_jsd(group):

            speed_without_nan = group['speed'].dropna()
            den_speed_without_nan = group['denoised_speed'].dropna()
            
            hist_speed, _ = np.histogram(
                speed_without_nan, bins=100, density=True
            )
            hist_denoised_speed, _ = np.histogram(
                den_speed_without_nan, bins=100, density=True
            )

            # Normalize the histograms to make sure they sum to 1
            hist_speed /= hist_speed.sum()
            hist_denoised_speed /= hist_denoised_speed.sum()

            jsd = jensenshannon(hist_speed, hist_denoised_speed, base=2)
            return jsd ** 2

        if window >= 0:
          self.traj_df["denoised_speed"] = (
              self.traj_df.groupby("person_id")["speed"]
              .rolling(window=window, center=False)
              .median()
              .reset_index(level=0, drop=True)
          )

        non_zero_den_speed = self.traj_df[
            (self.traj_df['denoised_speed'] != 0)
            & (self.traj_df['denoised_speed'].notna())].copy()

        den_speed_entropy = (non_zero_den_speed.groupby(
            'person_id')['denoised_speed']
            .apply(lambda x: entropy(x, base=2))
        )
        den_speed_std = (
            non_zero_den_speed.groupby("person_id")['denoised_speed'].std()
        )
        non_zero_speed = self.traj_df[
            (self.traj_df['speed'] != 0)
            & (self.traj_df['speed'].notna())].copy()   
                    
        speed_entropy = (non_zero_speed.groupby(
            'person_id')['speed']
            .apply(lambda x: entropy(x, base=2))
        )
        speed_std = non_zero_speed.groupby("person_id")['speed'].std()

        temp_df = pd.DataFrame({
            'person_id': speed_entropy.index,
            'speed_entr': speed_entropy.values,
            'speed_std': speed_std,
            'den_speed_entr': den_speed_entropy.values,
            'den_speed_std': den_speed_std,
            'x_std': speed_std - den_speed_std,
            'x_entr': speed_entropy.values - den_speed_entropy.values,
            'window': window,
        })
        
        jsd_values = non_zero_speed.groupby('person_id').apply(_calculate_jsd)
        temp_df['x_jsd'] = temp_df['person_id'].map(jsd_values) 

        if self.stats_df is None:
            self.stats_df = temp_df.copy()
        else:
            self.stats_df = pd.concat(
                [self.stats_df.copy(), temp_df.copy()], ignore_index=True
            )
            self.stats_df = self.stats_df.sort_values(
            by=["person_id", "window"]
            )

    def jsd_elbow_curve(self, min, max, step, plot=False):

        window_sizes = range(min, max + step, step)
        [self._make_stats(window_size) for window_size in window_sizes]

        self.elbow_points_df = pd.DataFrame()

        for person_id, group in self.stats_df.groupby('person_id'):
            x = group['window']
            y_jsd = group['x_jsd']

            if len(x) > 1 and len(y_jsd) > 1:
                slopes_jsd = np.diff(y_jsd) / np.diff(x)
                kneedle_jsd = KneeLocator(
                    x[:-1], slopes_jsd, curve='convex', direction='decreasing'
                )
                elbow_jsd = kneedle_jsd.elbow

                if elbow_jsd is not None:
                    elbow_jsd_index = np.where(x[:-1] == elbow_jsd)[0][0]
                    elbow_jsd_y = y_jsd.iloc[elbow_jsd_index]
                else:
                    elbow_jsd_y = np.nan

                # Plot if required
                if plot:
                    plt.figure(figsize=(12, 6))
                    plt.plot(x, y_jsd, label='JSD')
                    plt.xlabel('Window Size')
                    plt.ylabel('Jensen-Shannon Divergence')
                    plt.title(f'JSD Curve for {person_id}')
                    plt.legend()
                    plt.xticks(window_sizes)
                    plt.grid(True)
                    if elbow_jsd is not None:
                        plt.scatter(
                            elbow_jsd, elbow_jsd_y, 
                            color='red', label='Elbow Point'
                        )
                    plt.show()

                # Add the elbow point for JSD to the elbow_points_df
                self.elbow_points_df = self.elbow_points_df.append({
                    'person_id': person_id,
                    'jsd_elbow': elbow_jsd
                }, ignore_index=True)

            else:
                self.logger.info(
                    f"Not enough data points for {person_id}"
                    "to determine JSD elbow."
                )

        # Log the mean of the jsd_elbow points if there are any
        if not self.elbow_points_df.empty:
            mean_jsd_elbow = self.elbow_points_df['jsd_elbow'].mean()
            self.logger.info(
                f"Average JSD elbow across all individuals: {mean_jsd_elbow}"
            )
        else:
            self.logger.info("No JSD elbow points were calculated.")
        
        # Return the DataFrame with elbow points information
        return self.elbow_points_df
    
    def find_elbows(self, min, max, step, plot=False):
        window_sizes = range(min, max + step, step)
        [self._make_stats(window_size) for window_size in window_sizes]

        elbow_points_data = []

        for person_id, df_group in self.stats_df.groupby("person_id"):
            x = df_group['window']
            y1 = (-1 * df_group['x_entr']) 
            slopes1 = np.diff(y1) / np.diff(x)

            kneedle_first = KneeLocator(
                x[:-1], slopes1, curve='convex', direction='decreasing'
            )
            first_elbow = kneedle_first.elbow

            first_elbow_y = (
                y1[x == first_elbow].iloc[0]
                if first_elbow is not None 
                else np.nan
            )

            if first_elbow is not None:
                first_elbow_index = np.where(x == first_elbow)[0][0]
                x_new_segment = x[first_elbow_index + 1:]
                y1_new_segment = y1[first_elbow_index + 1:]
                slopes2 = np.diff(y1_new_segment) / np.diff(x_new_segment)
                
                if len(x_new_segment) > 1 and len(y1_new_segment) > 1:
                    kneedle_second = KneeLocator(
                        x_new_segment[:-1], slopes2, 
                        curve='convex', direction='decreasing'
                    )
                    second_elbow = kneedle_second.elbow
                    second_elbow_y = (
                        y1[x == second_elbow].iloc[0]
                        if second_elbow is not None 
                        else np.nan
                    )
                else:
                    second_elbow = None
                    second_elbow_y = np.nan
            else:
                second_elbow = None
                second_elbow_y = np.nan
            elbows = [
                val for val in [first_elbow, second_elbow]
                if val is not None
            ]
            mean_elbow = np.mean(elbows) if elbows else None

            elbow_points_data.append({
                'person_id': person_id,
                'first_elbow': first_elbow,
                'second_elbow': second_elbow,
                'mean_elbow': mean_elbow,
                'first_elbow_y': first_elbow_y,
                'second_elbow_y': second_elbow_y,
                'mean_x_entr': np.mean([first_elbow_y, second_elbow_y])
            })

            if plot:
                plt.figure(figsize=(12, 6))
                plt.plot(x, y1, label='Entropy Difference')
                plt.xlabel('Window Size')
                plt.ylabel('Difference')
                plt.title(f'Elbow Curve; {person_id}')
                plt.legend()
                plt.xticks(np.arange(min, max, step))
                plt.grid(True)
                if first_elbow is not None:
                    plt.scatter(
                        first_elbow, first_elbow_y, 
                        color='red', label='First Elbow Point'
                    )
                if second_elbow is not None:
                    plt.scatter(
                        second_elbow, second_elbow_y,
                        color='green', label='Second Elbow Point'
                    )
                plt.show()

        self.elbow_points_df = pd.DataFrame(elbow_points_data)
        
        self.logger.info("Suggested alphas: ")
        self.logger.info(
            f"First elbow: {self.elbow_points_df['first_elbow_y'].mean()}"
        )
        self.logger.info(
            f"Mid elbow: {self.elbow_points_df['mean_x_entr'].mean()}"
        )
        self.logger.info(
            f"Second elbow: {self.elbow_points_df['second_elbow_y'].mean()}"
        )

    def optimal_window_filter(
        self, alpha: Union[float, str] = 0.3, def_window=30):

        if alpha == 'auto':
            if 'person_id' in self.elbow_points_df.columns:
                self.elbow_points_df.set_index('person_id', inplace=True)
            
            self.x_entr_crossing_df = (
                self.elbow_points_df[['mean_elbow']]
                .rename(columns={'mean_elbow': 'window'})
            )
            self.x_entr_crossing_df['window'] = (
                self.x_entr_crossing_df['window']
                .fillna(def_window).astype(int)
            )
            for person_id in self.traj_df['person_id'].unique():
                if person_id in self.elbow_points_df.index:
                    window_size = (
                        self.elbow_points_df.loc[person_id, 'mean_elbow']
                    )
                    if pd.notnull(window_size):
                        denoised_speed = (
                            self.traj_df.loc[
                                self.traj_df['person_id'] == person_id, 'speed'
                            ].rolling(window=int(window_size), center=False)
                            .median()
                        )
                        self.traj_df.loc[
                            self.traj_df['person_id'] == person_id, 
                            'denoised_speed'
                            ] = denoised_speed        
        
        else:
            x_entr_crossing_data = []
            
            for person_id, df_group in self.stats_df.groupby("person_id"):
                x = df_group['window']
                y = df_group['x_std']
                y1 = -1 * df_group['x_entr']

                crossing_index = np.argmax(y1 >= alpha)
                crossing_window = (
                    x.iloc[crossing_index] 
                    if crossing_index < len(x) 
                    else None
                    )
                crossing_entr = (
                    -df_group['x_entr'].iloc[crossing_index]
                    if crossing_index < len(x)
                    else None
                    )

                x_entr_crossing_data.append({
                    'person_id': person_id,
                    'window': crossing_window,
                    'x_entr': crossing_entr
                })
            self.x_entr_crossing_df = pd.DataFrame(x_entr_crossing_data)
            self.x_entr_crossing_df = (
                pd.DataFrame(x_entr_crossing_data)
                .set_index('person_id')
                )
            self.x_entr_crossing_df['window'] = (
                self.x_entr_crossing_df['window']
                .fillna(def_window).astype(int)
                )
            for person_id in self.traj_df['person_id'].unique():
                if person_id in self.x_entr_crossing_df.index:
                    window_size = (
                        self.x_entr_crossing_df.loc[person_id, 'window']
                    )
                    if pd.notnull(window_size):
                        denoised_speed = (
                            self.traj_df.loc[
                                self.traj_df['person_id'] == person_id, 'speed'
                                ].rolling(window=window_size, center=False)
                                .median()
                        )
                    else:                
                        denoised_speed = (
                            self.traj_df.loc[
                                self.traj_df['person_id'] == person_id, 'speed'
                                ].rolling(window=def_window, center=False)
                                .median()
                        )
                    self.traj_df.loc[
                        self.traj_df['person_id'] == person_id,
                        'denoised_speed'
                        ] = denoised_speed

    def scipy_peaks_plot(self, real=False, tol=1.95):

        """
        1. Choose parameters looking at the graph per each user.
        2. Overlay extracted segments on the graph
        3. Set duration cut-off and export to .csv file

        """
        segments_found = False
        median_seg = False
        unique_person_ids = self.traj_df["person_id"].unique().tolist()

        dropdown = widgets.Dropdown(
            options=unique_person_ids,
            value=unique_person_ids[0] if unique_person_ids else None,
            description="Person ID:",
        )
        dropdown.layout = Layout(margin="15px 0 15px 0")
        window_size_slider = widgets.IntSlider(
            value=15,
            min=1,
            max=200,
            step=5,
            description="Window Size:",
            layout=widgets.Layout(width="50%"),
        )
        height_slider = widgets.FloatSlider(
            value=-0.8,
            min=-1.2,
            max=1.0,
            step=0.1,
            description="Height:",
            layout=widgets.Layout(width="40%"),
        )
        prom_slider = widgets.FloatSlider(
            value=None,
            min=-2.0,
            max=2.0,
            step=0.01,
            description="Prominence:",
            layout=widgets.Layout(width="40%"),
        )
        dist_slider = widgets.IntSlider(
            value=20,
            min=1,
            max=200,
            step=1,
            description="Distance:",
            layout=widgets.Layout(width="40%"),
        )
        wid_slider = widgets.IntSlider(
            value=10,
            min=1,
            max=100,
            step=1,
            description="Width:",
            layout=widgets.Layout(width="40%"),
        )
        height_slider1 = widgets.FloatSlider(
            value=0.3,
            min=-1.2,
            max=1.0,
            step=0.1,
            description="Height1:",
            layout=widgets.Layout(width="40%"),
        )
        prom_slider1 = widgets.FloatSlider(
            value=None,
            min=-2.0,
            max=2.0,
            step=0.01,
            description="Prominence1:",
            layout=widgets.Layout(width="40%"),
        )
        dist_slider1 = widgets.IntSlider(
            value=20,
            min=1,
            max=200,
            step=1,
            description="Distance1:",
            layout=widgets.Layout(width="40%"),
        )
        wid_slider1 = widgets.IntSlider(
            value=10,
            min=1,
            max=100,
            step=1,
            description="Width1:",
            layout=widgets.Layout(width="40%"),
        )

        # btn_find_segments = Button(description='Find Segments')
        btn_export_csv = Button(description="Export to CSV")
        btn_find_segments_med = Button(description="Find Segments")

        plot_output = Output()
        output = Output()

        duration_text = IntText(
            value=10,
            description="Duration:",
            disabled=False,
            style={"description_width": "initial"},
        )

        def _label_do():
            self.traj_df["label_estim"] = 0
            for _, segment in self.final_segments_df.iterrows():

                segment_range1 = (
                    (self.traj_df["person_id"] == segment["person_id"])
                    & (self.traj_df["n_point_id"] >= segment["start_id"])
                    & (self.traj_df["n_point_id"] <= segment["end_id"])
                )
                self.traj_df.loc[
                    segment_range1, "label_estim"
                    ] = segment["label"]

        def _on_export_csv_click(_):
            if self.final_segments_df is None:
                self.logger.info("Find the segments first!")
            else:
                self.export_segments_csv()

        def _on_find_segments_median_click(_):
            nonlocal segments_found
            nonlocal median_seg
            with output:
                clear_output()
                segments_found = True
                duration = max(0, duration_text.value)
                window_size = window_size_slider.value
                self.segment_graph("median", window_size)
                self.closest_exhibit_from_geom()
                self.club_segments_df(duration)
                self.refine_end_segment_boundaries(tol)
                # self.refine_start_segment_boundaries(2.1)
                _label_do()
                median_seg = True
                _update_plot()

        def _update_plot(change=None):
            nonlocal segments_found
            nonlocal median_seg
            with plot_output:
                plot_output.clear_output()
                selected_person_id = dropdown.value
                window_size = window_size_slider.value
                h = height_slider.value
                prom = prom_slider.value
                dist = dist_slider.value
                wid = wid_slider.value
                h1 = height_slider1.value
                p1 = prom_slider1.value
                d1 = dist_slider1.value
                w1 = wid_slider1.value

                # fig, ax = plt.subplots(figsize=(60, 10))
                # fig1, ax1 = plt.subplots(figsize=(60, 10))
                if real:
                    fig, (ax, ax1) = plt.subplots(2, 1, figsize=(60, 20))
                else:
                    fig = plt.figure(figsize=(60, 20))
                    ax = fig.add_subplot(1, 1, 1)
                if median_seg is False:
                    self.traj_df["denoised_speed"] = (
                        self.traj_df.groupby("person_id")["speed"]
                        .rolling(window=window_size, center=False)
                        .median()
                        .reset_index(level=0, drop=True)
                    )

                    selected_df = self.traj_df[
                        self.traj_df["person_id"] == selected_person_id
                    ]

                    x = selected_df["n_point_id"]
                    y = selected_df["denoised_speed"]
                    ax.plot(x, y, label="mov_med_speed", color="green")

                    if real:
                        if (self.groundtruth_df is not None):
                            ax1.plot(x, y, label="movMed_speed", color="green")                            
                            mask1 = selected_df["label_ground"] != 0
                            y2 = np.where(mask1, y, np.nan)
                            ax1.plot(x, y2, label="Ground truth segments", c="red")
                        else:
                            self.logger.info(
                                f"Ground truth was not initialized."
                            )                        
                    
                    peaks1 = scipy.signal.find_peaks(
                        -y, height=h,
                        distance=MovMedSpeedSegmenter.DEF_DIST,
                        width=MovMedSpeedSegmenter.DEF_WID,
                    )
                    peak_pos1 = peaks1[0]
                    heights1 = peaks1[1]["peak_heights"]

                    peaks2 = scipy.signal.find_peaks(
                        y, height=h1,
                        distance=MovMedSpeedSegmenter.DEF_DIST,
                        width=MovMedSpeedSegmenter.DEF_WID,
                    )
                    peak_pos2 = peaks2[0]
                    heights2 = peaks2[1]["peak_heights"]
                    ax.scatter(
                        peak_pos1,
                        -heights1,
                        color="blue",
                        s=30,
                        marker="x",
                        label="Valleys",
                    )
                    ax.scatter(
                        peak_pos2,
                        heights2,
                        color="red",
                        s=30,
                        marker="o",
                        label="Peaks",
                    )

                    if segments_found:
                        mask = selected_df["label"] != 0
                        y1 = np.where(mask, y, np.nan)
                        ax.plot(x, y1, label="Detected segments", c="#ffbe0b")

                else:
                    self.traj_df["denoised_speed"] = (
                        self.traj_df.groupby("person_id")["speed"]
                        .rolling(window=window_size, center=False)
                        .median()
                        .reset_index(level=0, drop=True)
                    )
                    selected_df = self.traj_df[
                        self.traj_df["person_id"] == selected_person_id
                    ]
                    x = selected_df["n_point_id"]
                    y = selected_df["denoised_speed"]
                    ax.plot(x, y, label="movMed_speed", color="green")


                    if real:
                        if (self.groundtruth_df is not None):
                            ax1.plot(x, y, label="movMed_speed", color="green")
                            mask1 = selected_df["label_ground"] != 0
                            y2 = np.where(mask1, y, np.nan)
                            ax1.plot(x, y2, label="Ground truth segments", c="red")
                        else:
                            self.logger.info(
                                f"Ground truth was not initialized."
                            )

                        self.red_peaks_ = self.traj_df.groupby("person_id").apply(
                            lambda x: scipy.signal.find_peaks(
                                x["denoised_speed"],
                                height=x["denoised_speed"].median(),
                                distance=MovMedSpeedSegmenter.DEF_DIST,
                                width=MovMedSpeedSegmenter.DEF_WID,)
                        )
                    
                    self.red_peaks = self.red_peaks_.apply(lambda x: x[0])
                    self.red_heights = self.red_peaks_.apply(lambda x: x[1])

                    self.blue_peaks_ = self.traj_df.groupby("person_id").apply(
                        lambda x: scipy.signal.find_peaks(
                            -x["denoised_speed"],
                            height=-np.median(
                                self.red_heights[x.name]["peak_heights"]
                                ),
                            distance=MovMedSpeedSegmenter.DEF_DIST,
                            width=MovMedSpeedSegmenter.DEF_WID,
                        )
                    )

                    self.blue_peaks = self.blue_peaks_.apply(lambda x: x[0])
                    self.blue_heights = self.blue_peaks_.apply(lambda x: x[1])
                    peak_pos_blue = self.blue_peaks[selected_person_id]
                    heights_blue = (
                        self.blue_heights[selected_person_id]["peak_heights"]
                        )

                    peak_pos_red = self.red_peaks[selected_person_id]
                    heights_red = (
                        self.red_heights[selected_person_id]["peak_heights"]
                    )

                    ax.scatter(
                        peak_pos_blue,
                        -heights_blue,
                        color="blue",
                        s=30,
                        marker="x",
                        label="Valleys",
                    )
                    ax.scatter(
                        peak_pos_red,
                        heights_red,
                        color="red",
                        s=30,
                        marker="o",
                        label="Peaks",
                    )
                    if segments_found:
                        mask = selected_df["label_estim"] != 0
                        y1 = np.where(
                            mask,
                            selected_df["denoised_speed"],
                            np.nan
                            )
                        ax.plot(x, y1, label="Detected segments", c="#ffbe0b")

                        if (self.red_heights is not None and
                                not self.red_heights.empty):
                            blue_line_y = np.median(self.red_heights[
                                selected_df["person_id"].iloc[0]
                                ]["peak_heights"]
                            )
                            red_line_y = selected_df["denoised_speed"].median()
                        else:
                            blue_line_y = 0
                            red_line_y = 0

                        ax.axhline(
                            y=blue_line_y,
                            color="blue",
                            linewidth=1.3,
                            label="Valleys threshold",
                        )
                        ax.axhline(
                            y=red_line_y,
                            color="red",
                            linewidth=1.3,
                            label="Peaks threshold",
                        )

                ax.set_xlabel("point_id")
                ax.set_ylabel("mov_med_speed")
                ax.set_title(
                    f'Moving median of speed. User: "{selected_person_id}" '
                    f'Window Size: "{window_size}"'
                )
                ax.legend()
                ax.grid(True, which="both", linestyle="--", linewidth=1)
                ax.set_xticks(np.arange(min(x), max(x) + 1, 150))

                if real:
                    ax1.set_xlabel("point_id")
                    ax1.set_ylabel("mov_med_speed")
                    ax1.set_title(
                        f"Ground truth segments"
                    )
                    ax1.legend()

                    ax1.grid(True, which="both", linestyle="--", linewidth=1)
                    ax1.set_xticks(np.arange(min(x), max(x) + 1, 150))

                plt.tight_layout()
                plt.show()

        duration_text.layout = Layout(margin="15px 0 15px 0")
        fndseg_layout = Layout(margin="15px 0 15px 0")
        btn_export_csv.layout = Layout(margin="5px 0 30px 0")
        btn_find_segments_med.layout = Layout(margin="5px 0 15px 0")
        btn_export_csv.on_click(_on_export_csv_click)
        btn_find_segments_med.on_click(_on_find_segments_median_click)
        # btn_find_segments.observe(_update_plot, names='value')  # type: ignore
        dropdown.observe(_update_plot, names="value")  # type: ignore
        window_size_slider.observe(_update_plot, names="value")  # type: ignore
        height_slider.observe(_update_plot, names="value")  # type: ignore
        prom_slider.observe(_update_plot, names="value")  # type: ignore
        dist_slider.observe(_update_plot, names="value")  # type: ignore
        wid_slider.observe(_update_plot, names="value")  # type: ignore
        height_slider1.observe(_update_plot, names="value")  # type: ignore
        prom_slider1.observe(_update_plot, names="value")  # type: ignore
        dist_slider1.observe(_update_plot, names="value")  # type: ignore
        wid_slider1.observe(_update_plot, names="value")  # type: ignore
        # heading1 = HTML(
        #     value="<h3 style='font-size:20px; font-weight:bold; color:red;'>"
        #     + "Adjust Peaks:</h3>"
        # )
        # heading2 = HTML(
        #     value="<h3 style='font-size:20px; font-weight:bold; color:#54c5e0;'>"
        #     + "Adjust Valleys:</h3>"
        # )

        blue_peaks_box = VBox([height_slider, prom_slider,
                               dist_slider, wid_slider])
        red_peaks_box = VBox([height_slider1, prom_slider1,
                              dist_slider1, wid_slider1])
        # display(VBox([heading1, red_peaks_box, heading2, blue_peaks_box,
        #               dropdown, window_size_slider, duration_text,
        #               btn_find_segments_med, output, btn_export_csv,
        #               plot_output])
        # )
        display(
            VBox(
                [
                    dropdown,
                    window_size_slider,
                    duration_text,
                    btn_find_segments_med,
                    output,
                    plot_output,
                ]
            )
        )
        _update_plot(None)

    def segment_graph(self, mode=None, window_size=None) -> pd.DataFrame:
        """
        1. Finds peaks and valleys in the moving median graph using
           Scipy's "find_peaks()":
           - high speed points (peaks): red_peaks_
           - low speed points (valleys): blue_peaks_ (by inverting the y-axis)
        2. Extracts segments: [peak, subsequent valley before next peak]
        3. Returns the segments_df
        """

        def _process_segment(
            df_group, start_index, end_index, person_id,
            label, s, valley_start, blu_line, red_line,) -> pd.DataFrame:

            """Calculates centroids, start, end points, duration of segments."""

            start_time = df_group.iloc[start_index]["time_stamp"]
            end_time = df_group.iloc[end_index]["time_stamp"]
            start_id = df_group.iloc[start_index]["n_point_id"]
            end_id = df_group.iloc[end_index]["n_point_id"]
            segment_df = df_group.iloc[start_index : end_index + 1]
            x_centroid = segment_df["x_kf"].mean()
            y_centroid = segment_df["y_kf"].mean()
            duration = (end_time - start_time).total_seconds()
            return [
                person_id, label, x_centroid, y_centroid, start_time,
                end_time, start_id, end_id, duration, s, valley_start,
                blu_line, red_line,
            ]

        if window_size == "auto":
            print("Finding segments... Window sizes = ")
            print(self.x_entr_crossing_df)
            if "denoised_speed" not in self.traj_df.columns:
                raise ValueError("'denoised_speed' must be calculated before \
                              calling segment_graph with window_size='auto'")
        else:
            if window_size is None:
                self.window_size = MovMedSpeedSegmenter.DEFAULT_WINDOW_SIZE
            else:
                self.window_size = window_size

        if (mode in ["median", None]) and (window_size != "auto"):
            self.traj_df["denoised_speed"] = (
                self.traj_df.groupby("person_id")["speed"]
                .rolling(window=self.window_size, center=False)
                .median()
                .reset_index(level=0, drop=True)
            )
        elif (mode == "mean") and (window_size != "auto"):
            self.traj_df["denoised_speed"] = (
                self.traj_df.groupby("person_id")["speed"]
                .rolling(window=self.window_size, center=False)
                .mean()
                .reset_index(level=0, drop=True)
            )
        elif (mode in ["ema", "ewm"]) and (window_size != "auto"):
            self.traj_df["denoised_speed"] = (
                self.traj_df.groupby("person_id", group_keys=True)["speed"]
                .apply(lambda x: x.ewm(
                    span=self.window_size, adjust=True
                    ).mean()
                    )
                .reset_index(level=0, drop=True)
            )

        self.red_peaks_ = self.traj_df.groupby("person_id").apply(
            lambda x: scipy.signal.find_peaks(
                x["denoised_speed"],
                height=x["denoised_speed"].median(),
                distance=MovMedSpeedSegmenter.DEF_DIST,
                width=MovMedSpeedSegmenter.DEF_WID,
            )
        )

        self.red_peaks = self.red_peaks_.apply(lambda x: x[0])
        self.red_heights = self.red_peaks_.apply(lambda x: x[1])
        self.blue_peaks_ = self.traj_df.groupby("person_id").apply(
            lambda x: scipy.signal.find_peaks(
                -x["denoised_speed"],
                height=-np.median(self.red_heights[x.name]["peak_heights"]),
                distance=MovMedSpeedSegmenter.DEF_DIST,
                width=MovMedSpeedSegmenter.DEF_WID,
            )
        )
        self.blue_peaks = self.blue_peaks_.apply(lambda x: x[0])
        self.blue_heights = self.blue_peaks_.apply(lambda x: x[1])

        # Extraction of segments
        self.logger.info(
            f"Finding segments... Window size = {window_size}"
        )
        segments = []
        for person_id, df_group in self.traj_df.groupby("person_id"):
            red_indices = self.red_peaks[person_id]
            blue_indices = self.blue_peaks[person_id]
            label = 1
            start_index = end_index = None
            blu_line = red_line = None
            valley_start = None
            s = 0.0
            first_point_id = df_group["n_point_id"].iloc[0]
            for i, row in enumerate(df_group.itertuples()):
                if i == first_point_id:
                    start_index = i
                if row.n_point_id in red_indices and start_index is None:
                    # Peak, start of a new segment
                    start_index = i
                elif row.n_point_id in blue_indices and start_index is not None:
                    # Valley after peak, possible end of the current segment
                    if valley_start is None:
                        valley_start = i
                    end_index = i
                elif row.n_point_id in red_indices and start_index is not None:
                    if end_index is not None:
                        # Next peak found
                        # Finalize the segment
                        blu_line = np.median(
                            self.red_heights[person_id]["peak_heights"]
                        )
                        red_line = np.median(row.denoised_speed)
                        s = df_group.iloc[
                            start_index : end_index
                            ]["denoised_speed"].std()

                        segments.append(
                            _process_segment(
                                df_group, start_index, end_index, person_id,
                                label, s, valley_start, blu_line, red_line,
                            )
                        )
                        label += 1
                        start_index = i
                        end_index = None
                        valley_start = None
                    else:
                        # Peak right after another peak, no valley
                        # Restart segment from present peak
                        start_index = i
                        end_index = None
            if start_index is not None and end_index is not None:
                segments.append(
                    _process_segment(
                        df_group, start_index, end_index, person_id, label,
                        s, valley_start, blu_line, red_line,
                    )
                )
        self.segments_df = pd.DataFrame(
            segments,
            columns=[
                "person_id", "label", "x_centroid", "y_centroid", "start_time",
                "end_time", "start_id", "end_id", "duration", "s",
                "valley_start", "blu_line", "red_line",
            ],
        )

        self.logger.info(
            f"Number of raw segments before merging: {len(self.segments_df)}"
        )
        return self.segments_df.copy()

    def closest_exhibit_from_point(self) -> pd.DataFrame:
        """
        Finds closest exhibit to each segment's centriod.
        For exhibit type: point coordinates.
        Returns the segments_df with closest exhibit and the distance to it.
        """

        self.segments_df = self.segments_df.sort_values(
            by=["person_id", "label"]
            )
        segments_array = self.segments_df[
            ["x_centroid", "y_centroid"]
            ].to_numpy()
        pois_array = self.exhibits_df[["x", "y"]].to_numpy()
        distances = cdist(segments_array, pois_array)
        closest_indices = distances.argmin(axis=1)
        self.segments_df["exhibit_o"] = (
            self.exhibits_df.iloc[closest_indices]["name"].values
        )
        self.segments_df["exh_dist"] = distances.min(axis=1)
        self.logger.info(
            "Found closest POIs"
        )
        return self.segments_df

    def closest_exhibit_from_geom(self, seg_df=None) -> pd.DataFrame:
        """
        Finds closest exhibit to each segment's centroid.
        For exhibit type: geometric objects.
        Returns the segments_df with closest exhibit and the distance to it.
        """
        if seg_df is not None:
            self.segments_df = seg_df

        self.segments_df = self.segments_df.sort_values(
            by=["person_id", "label"]
            )

        def _find_POI(row):
            point = Point(row["x_centroid"], row["y_centroid"])
            closest_exhibit = min(
                self.exhibits_df.itertuples(),
                key=lambda exhibit: point.distance(
                    loads(exhibit.geom, hex=True)
                    ),
            )

            distance = point.distance(loads(closest_exhibit.geom, hex=True))
            return pd.Series(
                [closest_exhibit.name, distance],
                index=["exhibit_o", "exh_dist"]
            )

        result_df = self.segments_df.apply(_find_POI, axis=1)
        self.segments_df[["exhibit_o", "exh_dist"]] = result_df
        self.logger.info(
            "Found closest POIs"
        )
        return self.segments_df

    def club_segments_df(self, 
                         cutoff_duration=None, seg_df=None) -> pd.DataFrame:
        """
        1. Clubs consecutive segments with same exhibit in proximity
        2. Optional: discards segments that have duration < cutoff_duration
        3. Recalculates centroids, updates start, end points and duration
           of segments.
        4. Returns the final_segments_df.
        """
        if seg_df is not None:
            self.segments_df = seg_df

        def _calculate_centroids(row):
            points_in_segment = self.traj_df[
                (self.traj_df["person_id"] == row["person_id"])
                & (self.traj_df["n_point_id"] >= row["start_id"])
                & (self.traj_df["n_point_id"] <= row["end_id"])
            ]
            return pd.Series(
                {
                    "x_centroid": points_in_segment["x_kf"].mean(),
                    "y_centroid": points_in_segment["y_kf"].mean(),
                }
            )

        def _calculate_std(row):
            relevant_data = self.traj_df[
                (self.traj_df["person_id"] == row["person_id"])
                & (self.traj_df["n_point_id"] >= row["valley_start"])
                & (self.traj_df["n_point_id"] <= row["end_id"])
            ]
            if len(relevant_data) >= 2:
                return relevant_data["denoised_speed"].std()
            else:
                return np.nan

        self.logger.info(
            "Merging consecutive segments with same POI..."
        )
        self.final_segments_df = (
            self.segments_df.merge(
                self.traj_df,
                left_on=["person_id", "start_id"],
                right_on=["person_id", "n_point_id"],
            )
            .groupby(
                [
                    "person_id",
                    "exhibit_o",
                    (
                        self.segments_df["exhibit_o"]
                        != self.segments_df["exhibit_o"].shift()
                    ).cumsum(),
                ]
            )
            .agg(
                person_id=("person_id", "first"),
                exhibit_o=("exhibit_o", "first"),
                start_time=("start_time", "first"),
                end_time=("end_time", "last"),
                start_id=("start_id", "first"),
                end_id=("end_id", "last"),
                exh_dist=("exh_dist", "mean"),
                s=("s", "first"),
                valley_start=("valley_start", "first"),
                blu_line=("blu_line", "first"),
                red_line=("red_line", "first"),
            )
            .reset_index(level=[0, 1], drop=True)
            .reset_index(drop=True)
        )
        self.final_segments_df = self.final_segments_df.sort_values(
            by=["person_id", "start_id"]
        )

        self.final_segments_df[
            ["x_centroid", "y_centroid"]
        ] = self.final_segments_df.apply(_calculate_centroids, axis=1)

        self.final_segments_df["s"] = self.final_segments_df.apply(
            _calculate_std, axis=1
        )

        self.final_segments_df = self.final_segments_df.assign(
            label=lambda x: x.groupby("person_id").cumcount() + 1,
            start_time=lambda x: pd.to_datetime(x["start_time"]),
            end_time=lambda x: pd.to_datetime(x["end_time"]),
            duration=lambda x: (
                x["end_time"] - x["start_time"]
                ).dt.total_seconds(),
        )

        if cutoff_duration is not None and cutoff_duration != 0:
            self.final_segments_df = self.final_segments_df[
                self.final_segments_df["duration"] >= cutoff_duration
            ]
            self.logger.info(f"Cut-off duration set: {cutoff_duration}")
        else:
            self.logger.info("No cut_off duration set")
        self.final_segments_df = self.final_segments_df.sort_values(
            by=["person_id", "start_id"]
        )
        self.final_segments_df = self.final_segments_df.assign(
            label=lambda x: x.groupby("person_id").cumcount() + 1,
        )
        self.logger.info(
            f"Final segments after merging: {len(self.final_segments_df)}"
        )

        return self.final_segments_df.copy()

    def refine_start_segment_boundaries(
            self, tolerance,
            fin_seg_df=None):

        if fin_seg_df is not None:
            self.final_segments_df = fin_seg_df.copy()

        self.final_segments_df["next_start_id"] = (
            self.final_segments_df["start_id"]
            .shift(-1)
            .ffill()
            .astype(int)
        )

        for index, row in self.final_segments_df.iterrows():
            if row["valley_start"] == row["end_id"]:
                continue
            person_id = row["person_id"]
            traj_data = self.traj_df[self.traj_df["person_id"] == person_id]
            blu = np.median(
                self.red_heights[traj_data["person_id"].iloc[0]]
                ["peak_heights"]
            )
            for i in range(row["start_id"], row["valley_start"]):
                current_speed = traj_data.loc[
                    traj_data["n_point_id"] == i, "denoised_speed"
                ].iloc[0]
                valley_speed = traj_data.loc[
                    traj_data["n_point_id"] == row["valley_start"],
                    "denoised_speed"
                ].iloc[0]
                if current_speed > blu:
                    if current_speed < valley_speed + row["s"] * tolerance:
                        time = traj_data.loc[
                            traj_data["n_point_id"] == i, "time_stamp"
                        ].iloc[0]
                        self.final_segments_df.at[index, "start_id"] = i
                        self.final_segments_df.at[index, "start_time"] = time
                        break
        self.logger.info("Starts adjusted.")
        return self.final_segments_df.copy()

    def refine_end_segment_boundaries(
            self, tolerance,
            fin_seg_df=None):

        self.logger.info(
            f"Refining ends with tolerance = {tolerance}..."
        )
        if fin_seg_df is not None:
            self.final_segments_df = fin_seg_df.copy()

        self.final_segments_df["next_start_id"] = (
            self.final_segments_df["start_id"]
            .shift(-1)
            .ffill()
            .astype(int)
        )

        for index, row in self.final_segments_df.iterrows():
            if row["valley_start"] == row["end_id"]:
                continue
            person_id = row["person_id"]
            traj_data = self.traj_df[self.traj_df["person_id"] == person_id]
            for i in range(row["end_id"], row["next_start_id"]):
                current_speed = traj_data.loc[
                    traj_data["n_point_id"] == i, "denoised_speed"
                ].iloc[0]
                end_speed = traj_data.loc[
                    traj_data["n_point_id"] == row["end_id"], "denoised_speed"
                ].iloc[0]
                if current_speed > end_speed + (row["s"] * tolerance):
                    time = traj_data.loc[
                        traj_data["n_point_id"] == i, "time_stamp"
                    ].iloc[0]
                    self.final_segments_df.at[index, "end_id"] = i
                    self.final_segments_df.at[index, "end_time"] = time
                    break
        self.logger.info(
            f"Done! ✨🍰✨"
        )
        return self.final_segments_df.copy()

    def export_segments_csv(self, path=None):
        if self.final_segments_df is None:
            self.logger.info("Segments empty. Nothing to write.")
        if path is not None:
            self.final_segments_df.to_csv(path, index=True)
            self.logger.info(f"Exported to {path}!")
        else:
            self.final_segments_df.to_csv("/content/final_segments.csv",
                                          index=True)
            self.logger.info("Exported to /content/final_segments.csv!")
        