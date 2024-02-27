tol = 2.00

# Pass the ground truth dataframe (here "df_gt") in the constructor
object2 = MovMedSpeedSegmenter(df, exhibits_df, df_gt)
object2.process_trajectory_data()

object2.scipy_peaks_plot(real=True, tol=tol)