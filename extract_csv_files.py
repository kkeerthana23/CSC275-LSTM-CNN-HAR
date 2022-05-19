import numpy as np
import pandas as pd

file = "/Users/vamshireddy/Downloads/SPRING2022/CSC275/CSC275-LSTM-CNN-HAR/input_files/X_bed_win_1000_thrshd_60percent_step_200.npz"
file1 = "/Users/vamshireddy/Downloads/SPRING2022/CSC275/CSC275-LSTM-CNN-HAR/input_files/X_bed_win_1000_thrshd_60percent_step_200_final.csv"
data_bed_x = np.load(file)['arr_0']

np.savetxt(file1, data_bed_x, delimiter=",")

#pd.DataFrame(np_array).to_csv()
print("done")


