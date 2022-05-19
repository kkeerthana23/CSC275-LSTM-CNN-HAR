import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

path = "/Users/vamshireddy/Downloads/SPRING2022/CSC275/CSC275-LSTM-CNN-HAR/Dataset/Data/input_161219_siamak_"


input_data = pd.read_csv(path+"bed"+"_1.csv",header=None).values

amp = input_data[:, 1:91]

# plt
fig = plt.figure(figsize=(10, 10))
ax1 = plt.subplot(311)
plt.imshow(amp[:, 0:29].T, interpolation="nearest", aspect="auto", cmap="jet")
ax1.set_title("Antenna1 Amplitude bed")
plt.colorbar()

plt.figure()

input_data = pd.read_csv(path+"fall"+"_1.csv",header=None).values

amp = input_data[:, 1:91] 

ax1 = plt.subplot(311)
plt.imshow(amp[:, 0:29].T, interpolation="nearest", aspect="auto", cmap="jet")
ax1.set_title("Antenna1 Amplitude fall")
plt.colorbar()

plt.figure()

input_data = pd.read_csv(path+"pickup"+"_1.csv",header=None).values

amp = input_data[:, 1:91]

ax1 = plt.subplot(311)
plt.imshow(amp[:, 0:29].T, interpolation="nearest", aspect="auto", cmap="jet")
ax1.set_title("Antenna1 Amplitude pick")
plt.colorbar()

plt.figure()

input_data = pd.read_csv(path+"run"+"_1.csv",header=None).values

amp = input_data[:, 1:91]

ax1 = plt.subplot(311)
plt.imshow(amp[:, 0:29].T, interpolation="nearest", aspect="auto", cmap="jet")
ax1.set_title("Antenna1 Amplitude run")
plt.colorbar()

plt.figure()

input_data = pd.read_csv(path+"sitdown"+"_1.csv",header=None).values

amp = input_data[:, 1:91]


ax1 = plt.subplot(311)
plt.imshow(amp[:, 0:29].T, interpolation="nearest", aspect="auto", cmap="jet")
ax1.set_title("Antenna1 Amplitude sitdown")
plt.colorbar()

plt.figure()

input_data = pd.read_csv(path+"standup"+"_1.csv",header=None).values

amp = input_data[:, 1:91]

ax1 = plt.subplot(311)
plt.imshow(amp[:, 0:29].T, interpolation="nearest", aspect="auto", cmap="jet")
ax1.set_title("Antenna1 Amplitude standup")
plt.colorbar()

plt.figure()

input_data = pd.read_csv(path+"walk"+"_1.csv",header=None).values

amp = input_data[:, 1:91]

ax1 = plt.subplot(311)
plt.imshow(amp[:, 0:29].T, interpolation="nearest", aspect="auto", cmap="jet")
ax1.set_title("Antenna1 Amplitude walk")
plt.colorbar()

plt.figure()

plt.show()