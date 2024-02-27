import logging
import math
import os
import time
from datetime import datetime

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import scipy.signal
from ipywidgets import HTML, VBox, Layout, HBox, Label, Button, Output, IntText
import ipywidgets as widgets
from IPython.display import display, clear_output
from kneed import KneeLocator
from pulp import *
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist, jensenshannon
from scipy.stats import entropy
from shapely.geometry import Point, MultiLineString
from shapely.wkb import loads


class Optimizer:
    """
    1. Designed to optimize parameters for the MovMedSpeedSegmenter by
    evaluating different configurations against ground truth data. 
    2. Aims to determine the optimal window sizes and tolerance levels for 
    segmenting trajectories and matching them to the ground truth.
    
    Attributes:
        a (int): Placeholder attribute, can be used for further extensions or
                 configurations.
        all_eval_results (pd.DataFrame): Stores evaluation results for various
                 configurations during window size optimization.
        tol_results_df (pd.DataFrame): Stores evaluation results for various
                 tolerance levels during tolerance optimization.
    """    

    def __init__(self):
        self.a = 0

    def optimize_window(
            self, win_range=(1, 100, 3) , dur=10, tol=2.00, plot=False):
        """
        1. Optimizes the window size of the moving median.        
        2. Iterates over a range of window sizes to find the one that yields
            the best S-score compared to ground truth data.
        
        Parameters:
            win_range (tuple): The range and step of window sizes to test
                               (min, max, step).
            dur (int): Minimum duration for a segment to be considered valid.
            tol (float): Tolerance for refining segment end boundaries.
            plot (bool): If True, plots the S-Score for each window size.
        """

        min, max, step = win_range
        tol = tol
        cutoff_duration = dur

        seg_med = MovMedSpeedSegmenter(df, exhibits_df)
        seg_med.process_trajectory_data()

        all_results = []

        for win in range(min, max, step):
            seg_med.segment_graph(mode="median", window_size=win)
            seg_med.closest_POI_from_geom() 
            bef_df = seg_med.merge_segments_df(cutoff_duration)
            result_segments = seg_med.refine_end_segment_boundaries(tol, bef_df)

            eval_med = MatchingRetrieverNew(result_segments, df_gt)
            eval_med.process_prep()
            eval_results = eval_med.find_match(alfa=0.5)

            all_results.append(eval_results.copy())
        self.all_eval_results = pd.concat(all_results, ignore_index=True)

        clear_output(wait=True)
        print(f"Highest S-scores: ")
        for person_id, group in self.all_eval_results.groupby('person_id'):
            max_jacc_row = group.loc[group['avg_jacc'].idxmax()]
            print(
                f"{person_id}: {max_jacc_row['avg_jacc']:.2f} \
                at {max_jacc_row['window']}"
                )

        print(f"Highest F-scores: ")
        for person_id, group in self.all_eval_results.groupby('person_id'):
            max_jacc_row = group.loc[group['f_measure'].idxmax()]
            print(
                f"{person_id}: {max_jacc_row['f_measure']:.2f} \
                at {max_jacc_row['window']}"
                )

        if plot:
            for person_id in self.all_eval_results['person_id'].unique():
                person_results = self.all_eval_results[
                    self.all_eval_results['person_id'] == person_id
                    ]
                person_results = person_results.sort_values(by='window')
                
                plt.figure(figsize=(10, 6))
                plt.plot(
                    person_results['window'],
                    person_results['avg_jacc'],
                    marker='o', linestyle='-'
                    )
                plt.title(f'Person ID: {person_id}')
                plt.xlabel('Window')
                plt.ylabel('S-Score')
                plt.grid(True)
                plt.show()

    def optimize_end_tolerance(self, tol_range=(10, 60, 0.05)):
        """
        1. Optimizes the end segment boundary refinement tolerance.
        2. Iterates over a range of tolerance values to find the one that best
            refines segment boundaries according to ground truth data.
        
        Parameters:
            tol_range (tuple): The range and step of tolerance values to test
                               (min, max, step).
        """

        self.tol_results_df = pd.DataFrame()  # Initialize the DataFrame        
        min, max, step = tol_range
        mode = "median"
        cutoff_duration = 10
        thresholds = [i * step for i in range(min, max)]

        columns = ["threshold", "total_mean", "total_std_dev"]
        metrics = ['f_measure', 'precision', 'avg_jacc']
        for metric in metrics:
            columns.append(metric + "_mean")
            columns.append(metric + "_std_dev")

        results_df = pd.DataFrame(columns=columns)

        o_current = MovMedSpeedSegmenter(df, exhibits_df, df_gt)
        o_current.process_trajectory_data()

        o_current.find_elbows(min=1, max=151, step=3, plot=False)
        o_current.optimal_window_filter(alpha='auto')

        o_current.segment_graph(mode="median", window_size="auto")
        o_current.closest_POI_from_geom()
        bef_df = o_current.merge_segments_df(cutoff_duration)

        for threshold in thresholds:
            start_time = time()

            print(f"Threshold: {threshold}")
            fin_seg_df_ = o_current.refine_end_segment_boundaries(threshold, bef_df)
            
            eval_current = MatchingRetrieverNew(fin_seg_df_, df_gt)
            eval_current.process_prep()
            eval_current.find_match(alfa=0.5)

            metrics_mean = eval_current.results_df[metrics].mean()
            metrics_std_dev = eval_current.results_df[metrics].std()
            total_mean = metrics_mean.sum()
            total_std_dev = metrics_std_dev.sum()
            
            end_time = time()
            total_time = end_time - start_time

            tp = eval_current.results_df['true_positive'].sum()            
            result_dict = {
                "threshold": threshold,
                "total_mean": total_mean,
                "total_std_dev": total_std_dev,
                "execution_time": total_time,
                "tp": tp,
                }
            for metric in metrics:
                result_dict[metric + "_mean"] = metrics_mean[metric]
                result_dict[metric + "_std_dev"] = metrics_std_dev[metric]

            self.tol_results_df = pd.concat(
                [self.tol_results_df, pd.DataFrame([result_dict])], ignore_index=True
                )
        
        clear_output(wait=True)
        self.tol_results_df.reset_index(drop=True, inplace=True)
        print("Total highest mean(F, S): ")
        print(self.tol_results_df.loc[
            self.tol_results_df["total_mean"].idxmax(),
            ['threshold', 'total_mean', 'f_measure_mean',
            'avg_jacc_mean', 'tp']
            ])

    def optimize_entr_diff(self, win_range=(1, 150, 3)):
        min, max, step = win_range
        cutoff_duration = 10

        user_scores_by_window = {}
        best_window_for_f_measure = {}
        best_window_for_avg_jacc = {}

        o_current1 = MovMedSpeedSegmenter(df, exhibits_df, df_gt)
        o_current1.process_trajectory_data()
        o_current1.find_elbows(min=min, max=max, step=step, plot=False)
        
        for window_length in range(min, max, step):

            o_current1.segment_graph("median", window_length)
            o_current1.closest_POI_from_geom()
            o_current1.merge_segments_df(cutoff_duration)
            o_current1.merge_segments_df(cutoff_duration)
            
            eval_current1 = MatchingRetrieverNew(o_current1.final_segments_df, df_gt)
            eval_current1.process_prep()
            eval_current1.find_match(alfa=0.5)

            eval_current1.results_df = pd.merge(
                eval_current1.results_df.copy(),
                o_current1.stats_df[['person_id','x_std', 'x_entr']].round(1),
                on='person_id', how='left'
            ).copy()

            user_scores_by_window[window_length] = eval_current1.results_df.copy()

        for window_length, scores_df in user_scores_by_window.items():
            for index, row in scores_df.iterrows():
                user_id = row['person_id']
                f_measure = row['f_measure']

                if (
                    user_id not in best_window_for_f_measure 
                    or f_measure > best_window_for_f_measure[user_id]['f_measure']
                    ):
                    best_window_for_f_measure[user_id] = {
                        'window_length': window_length,
                        'f_measure': f_measure,
                        'x_std': row['x_std'],
                        'x_entr': row['x_entr'],
                        }

        for window_length, scores_df in user_scores_by_window.items():
            for index, row in scores_df.iterrows():
                user_id = row['person_id']
                avg_jacc = row['avg_jacc']

                if (
                    user_id not in best_window_for_avg_jacc
                    or avg_jacc > best_window_for_avg_jacc[user_id]['avg_jacc']
                    ):
                    best_window_for_avg_jacc[user_id] = {
                        'window_length': window_length,
                        'avg_jacc': avg_jacc,
                        'x_std': row['x_std'],
                        'x_entr': row['x_entr'],
                        }

        person_ids = []
        window_lengths = []
        f_measures = []
        x_stds = []
        x_entrs = []

        for user_id, best_config in best_window_for_f_measure.items():
            person_ids.append(user_id)
            window_lengths.append(best_config['window_length'])
            f_measures.append(best_config['f_measure'])
            x_stds.append(best_config['x_std'])
            x_entrs.append(best_config['x_entr'])

        self.df_f_measure = pd.DataFrame({
            'person_id': person_ids,
            'window_length': window_lengths,
            'f_measure': f_measures,
            'x_std': x_stds,
            'x_entr': x_entrs
        })

        person_ids = []
        window_lengths = []
        avg_jaccs = []
        x_stds = []
        x_entrs = []

        for user_id, best_config in best_window_for_avg_jacc.items():
            person_ids.append(user_id)
            window_lengths.append(best_config['window_length'])
            avg_jaccs.append(best_config['avg_jacc'])
            x_stds.append(best_config['x_std'])
            x_entrs.append(best_config['x_entr'])

        self.df_avg_jacc = pd.DataFrame({
            'person_id': person_ids,
            'window_length': window_lengths,
            'avg_jacc': avg_jaccs,
            'x_std': x_stds,
            'x_entr': x_entrs
        })
        clear_output(wait=True)
        print("Entropy difference at max S-score: ")
        print(df_avg_jacc)
        print("Entropy difference at max F-score: ")
        print(df_avg_jacc)