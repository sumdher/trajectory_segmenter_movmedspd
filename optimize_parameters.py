from src.optimizer import Optimizer

opt = Optimizer()

# Finds segments and evaluates them by trying a range(win_range) of windows for the moving median.
# If plot=True, it plots the evaluation result - S-score as window increases, for each trajectory.
# The other arguments are the parameters for the methods in the MovMedSpeedSegmenter.
opt.optimize_window(win_range=(1, 100, 3) , dur=10, tol=2.00, plot=False)


# This shows which tolerance parameter for terminal point refinement yields the maximum evaluation results.
opt.optimize_end_tolerance(tol_range=(10, 60, 0.05))

# This automatically chooses the window sizes for each trajectory based on the entropy difference before and after applying the filter.
# This shows for which entropy difference the evaluation results are maximum.
opt.optimize_entr_diff(win_range=(1, 150, 3))

