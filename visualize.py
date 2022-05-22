import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

path = "/Users/vamshireddy/Downloads/SPRING2022/CSC275/CSC275-LSTM-CNN-HAR/Dataset/Data/input_161219_siamak_"


input_data = pd.read_csv(path+"walk"+"_1.csv",header=None).values

amp = input_data[:, 1:91]

# =============================================================================
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
# =============================================================================



# =============================================================================
# def moving_average(data, window_size):
#     window= np.ones(int(window_size))/float(window_size)
#     return np.convolve(data, window, 'same')
# 
# # Initializing valiables
# constant_offset = np.empty_like(amp)
# filtered_data = np.empty_like(amp)
# 
# # Calculating the constant offset (moving average 4 seconds)
# for i in range(1, len(amp[0])):
#     constant_offset[:,i] = moving_average(amp[:,i], 4000)
# 
# # Calculating the filtered data (substract the constant offset)
# filtered_data = amp - constant_offset
# 
# # Smoothing (moving average 0.01 seconds)
# for i in range(1, len(amp[0])):
#     filtered_data[:,i] = moving_average(filtered_data[:,i], 10)
# # Calculate correlation matrix (90 * 90 dim)
# cov_mat2 = np.cov(filtered_data.T)
# # Calculate eig_val & eig_vec
# eig_val2, eig_vec2 = np.linalg.eig(cov_mat2)
# # Sort the eig_val & eig_vec
# idx = eig_val2.argsort()[::-1]
# eig_val2 = eig_val2[idx]
# eig_vec2 = eig_vec2[:,idx]
# # Calculate H * eig_vec
# pca_data2 = filtered_data.dot(eig_vec2)
# 
# 
# plt.figure(figsize = (18,30))
# # Spectrogram(STFT)
# plt.subplot(611)
# Pxx, freqs, bins, im = plt.specgram(pca_data2[:,6], NFFT=128, Fs=1000, noverlap=1, cmap="jet", vmin=-100,vmax=20)
# plt.xlabel("Time[s]")
# plt.ylabel("Frequency [Hz]")
# plt.title("Spectrogram(STFT)")
# plt.colorbar(im)
# plt.xlim(0,10)
# plt.ylim(0,100)
# =============================================================================























