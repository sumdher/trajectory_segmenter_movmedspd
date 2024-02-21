# Segmenter MovMedSpd
- Detects low speed segments from spatiotemporal trajectories and determines the closest Point Of Interest (POI) from each segment's centroid.
- Extracts, processes and works on the main feature: _speed_. Applies a denoising filter: Moving median (on speed).
- Problem: Discrete segmentation of a time-series signal, signal processing.

Key liraries:
_[find_peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html)_ to find the peaks and valleys.
_[KneeLocator](https://pypi.org/project/kneed/)_ to find the elbow points.
_[ipywidgets](https://ipywidgets.readthedocs.io/en/stable/)_ for a Google Colab/Jupyter interface for dynamic visualization.

## Strengths:
- Data-driven, adaptive determination of window sizes for moving median from the data, instead of arbitrarily setting it. (Entropy difference and Jensen-Shannon Divergence)
- Interface for dynamic visualiations (of graphs) to manually determine and define the parameters.
- Comparable results with existing algorithms; sometimes better in some aspects. (See section: "**Results**")
- Fast and robust.

## Weaknesses:
- Unusual behaviour if the data is too noisy.
- Needs POIs' locations to be known in advance (Future work).

## Requires:
**Spatiotemporal trajactory data**

Shape: (x, y) coordinates and timestamps, unique trajectory ID for multiple trajectories.

**POI data**

Shape: (x, y) coordinates, unique ID/name.

**[OPTIONAL] Ground truth stops/segments** (Only for evaluation and testing)

Shape: start_time, end_time, (x, y) centroid, closest POI.

## Output:
**Segments**

Shape: start_time, end_time, (x, y) centroid, closest POI (and other things)

# Results



# Screengraps of the Interface

![image](https://github.com/sumdher/MovMedSpdEval/assets/26754139/7d983d04-8dbb-4e41-beef-9d9109d19f02)
![image](https://github.com/sumdher/MovMedSpdEval/assets/26754139/802dacfd-72ad-43a9-a9b3-fea0da9b9105)



