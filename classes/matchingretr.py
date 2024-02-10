
import math
import numpy as np
import pandas as pd
from pulp import *
# from pulp import PULP_CBC_CMD


class MatchingRetrieverNew:
    """Evalueates the segments against the ground truth segments.
    Calculates various scores: F-score, S-score, Precision, Recall, Accuracy,
    TP, TN, FP, FN. 
    
    Determines the temporal overlap of the detected (result_segmets) with the 
    real ground truth segments. Macthes them, finds the amount of overlap along
    with the spatial distance between the centroids of detected and the real
    segments. Most relevant measures: f_measure, avg_jacc.

    Matches detected segments with real ones. `alfa` is the weight in the
    calculation of avg_jacc, which is a weighted average of temporal overlap
    and the spatial distance between the centroids of the detected and real
    segments.
    

    """
    
    J_LIMIT = 0.0

    def __init__(self, cl_df, gt_df):
        self.duration = 10.0
        self.alfa = 0.5
        self.beta = 1 - self.alfa        
        self.cl_df = cl_df.copy()
        self.gt_df = gt_df.copy()
        self.gt_lists = None
        self.cl_lists = None
        self.results_df = None

    def process_prep(self):

        self.cl_df.sort_values(by=["person_id", "label"], inplace=True)
        self.gt_df.sort_values(by=["person_id", "gt_labeling"], inplace=True)

        self.cl_df["start_time"] = pd.to_datetime(self.cl_df["start_time"])
        self.cl_df["end_time"] = pd.to_datetime(self.cl_df["end_time"])
        self.gt_df["start"] = pd.to_datetime(self.gt_df["start"])
        self.gt_df["end"] = pd.to_datetime(self.gt_df["end"])

        self.cl_df["start_time_seconds"] = self.cl_df["start_time"].apply(
            lambda x: x.timestamp()
        )
        self.cl_df["end_time_seconds"] = self.cl_df["end_time"].apply(
            lambda x: x.timestamp()
        )

        self.gt_df["start_time_seconds"] = self.gt_df["start"].apply(
            lambda x: x.timestamp()
        )
        self.gt_df["end_time_seconds"] = self.gt_df["end"].apply(
            lambda x: x.timestamp()
        )

        self.gt_lists = []
        for person_id in self.gt_df["person_id"].unique():
            filtered_df = self.gt_df[self.gt_df["person_id"] == person_id]
            current_list = filtered_df.apply(
                lambda row: [
                    row["gt_labeling"],
                    row["start_time_seconds"],
                    row["end_time_seconds"],
                    round((row["end_time_seconds"]
                         - row["start_time_seconds"]), 3),
                    (row["x_centroid"],
                    row["y_centroid"]),
                    row["person_id"],
                    row["gt_poi"],
                ],
                axis=1,
            ).tolist()
            self.gt_lists.append(current_list)

        self.cl_lists = []

        for person_id in self.cl_df["person_id"].unique():
            filtered_df = self.cl_df[self.cl_df["person_id"] == person_id]
            current_list = filtered_df.apply(
                lambda row: [
                    row["label"],
                    row["start_time_seconds"],
                    row["end_time_seconds"],
                    round((
                        row["end_time_seconds"]
                      - row["start_time_seconds"]), 3),
                    (row["x_centroid"],
                    row["y_centroid"]),
                    row["person_id"],
                    row["exhibit_o"],
                ],
                axis=1,
            ).tolist()
            self.cl_lists.append(current_list)

    def find_match(self, alfa=0.5):

        if alfa is not None:
            self.alfa = alfa
            self.beta = 1 - self.alfa
        results_list = []
        for gt, cl in zip(self.gt_lists, self.cl_lists):
            self.long_stops_count = 0
            self.short_stops_count = 0
            long_gt = []
            short_gt = []
            for g in gt:
                if g[3] >= self.duration:
                    self.long_stops_count += 1
                    long_gt.append(g)
                else:
                    self.short_stops_count += 1
                    short_gt.append(g)

            gt = long_gt

            all_stops = [
                ("S"
                 + str(item[0]), item[1], item[2], item[3], item[4], item[5])
                for item in gt
            ]
            all_clusters = [
                (str(item[0]), item[1], item[2], item[3], item[4], item[5])
                for item in cl
            ]
            length = max(len(gt), len(cl))

            if len(gt) < len(cl):
                diff = len(cl) - len(gt)
                for i in range(diff):
                    all_stops.append(
                        ("SNULL", -1, -1, -1, ("NULL", "NULL"), -1)
                        )
            elif len(cl) < len(gt):
                diff = len(gt) - len(cl)
                for i in range(diff):
                    all_clusters.append(
                        ("CNULL", -1, -1, -1, ("NULL", "NULL"), -1)
                        )

            intersection_doublearray = []

            stops_set = set()
            clusters_set = set()
            max_dist = 0.0

            for s in all_stops:
                l = []
                for c in all_clusters:
                    if (s[1] <= c[1] and c[1] <= s[2]) or (
                        c[1] <= s[1] and s[1] <= c[2]
                    ):
                        if s[5] == c[5]:
                            intersection = min(c[2], s[2]) - max(c[1], s[1])
                            union = max(c[2], s[2]) - min(c[1], s[1])
                            j = float(intersection) / float(union)

                            if (
                                j * 100 > MatchingRetrieverNew.J_LIMIT
                            ):
                                dist = math.dist(s[4], c[4])
                                if self.alfa == 0:
                                    l.append((j, dist))
                                elif self.alfa == 1:
                                    l.append((j, dist))
                                else:
                                    l.append((j, dist))

                                stops_set.add(s[0])
                                clusters_set.add(c[0])

                            else:
                                l.append((0, 0))
                        else:
                            l.append((0, 0))
                    else:
                        l.append((0, 0))
                intersection_doublearray.append(l)

            l_alfa = []
            for row in intersection_doublearray:
                l_alfa_row = []
                for ts in row:
                    if ts[0] > 0:
                        l_alfa_row.append(
                            self.alfa*ts[0] + self.beta*(1/(1+ts[1]))
                        )
                    else:
                        l_alfa_row.append(0.0)
                l_alfa.append(l_alfa_row)

            intersection_matrix = np.array(l_alfa)

            # print(intersection_matrix)
            clusters = list(clusters_set)
            clusters.sort()
            stops = list(stops_set)
            stops.sort()

            prob = LpProblem("Matching_stops", LpMaximize)
            y = LpVariable.dicts(
                "pair",
                [(i, j) for i in range(length) for j in range(length)],
                cat="Binary",
            )
            prob += lpSum(
                [
                    (intersection_matrix[i][j]) * y[(i, j)]
                    for i in range(length)
                    for j in range(length)
                ]
            )

            for i in range(length):
                prob += lpSum(y[(i, j)] for j in range(length)) <= 1
                prob += lpSum(y[(j, i)] for j in range(length)) <= 1
                prob += lpSum(y[(i, j)] + y[(j, i)] for j in range(length)) == 2
            prob += (
                lpSum(y[(i, j)]
                for i in range(length)
                for j in range(length)) == length
            )

            # prob.solve()

            prob.solve(PULP_CBC_CMD(msg=False))
            
            average_j = 0.0
            true_positive = 0
            false_negative = 0
            false_positive = 0
            true_negative = 0
            intersection_short = 0
            matched_avgs = []
            sum_avg = 0.0

            true_clusters = []
            true_detected_stops = []

            for i in range(length):
                for j in range(length):
                    if y[(i, j)].varValue == 1:
                        if intersection_matrix[i, j] > 0:
                            if all_stops[i][3] >= self.duration:
                                average_j += intersection_matrix[i, j]
                                true_positive += 1
                                true_clusters.append(all_clusters[j])
                                true_detected_stops.append(all_stops[i])
                                intersection = min(
                                    all_clusters[j][2], all_stops[i][2]
                                ) - max(all_clusters[j][1], all_stops[i][1]
                                )
                                union = max(all_clusters[j][2], all_stops[i][2]
                                )- min(all_clusters[j][1], all_stops[i][1]
                                )
                                jacc = float(intersection) / float(union)
                                matched_avgs.append(
                                    (all_stops[i], all_clusters[j], jacc)
                                )
                                sum_avg += jacc

                            elif all_stops[i][3] > 0:
                                intersection_short += 1

            true_negative = self.short_stops_count - intersection_short
            false_positive = len(cl) - true_positive
            false_negative = self.long_stops_count - true_positive

            if true_positive > 0:
                avg_jacc = sum_avg / len(matched_avgs)
                average_j = average_j / (100 * true_positive)

                recall = true_positive / (true_positive + false_negative)
                precision = true_positive / (true_positive + false_positive)
                f_measure = 2 * recall * precision / (recall + precision)
                # print(f_measure)

            else:
                avg_jacc = 0
                average_j = 0
                recall = 0
                precision = 0
                f_measure = 0
            # print(f_measure)

            accuracy = float(true_positive + true_negative) / float(
                true_positive + true_negative + false_positive + false_negative
            )

            new_row = {
                "accuracy": accuracy,
                "true_positive": true_positive,
                "true_negative": true_negative,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "f_measure": f_measure,
                "recall": recall,
                "precision": precision,
                "avg_jacc": average_j * 100,
                "person_id": gt[0][-2],
            }

            results_list.append(new_row)
        self.results_df = pd.DataFrame(results_list)
        return self.results_df.copy()
