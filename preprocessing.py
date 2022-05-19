import glob
import numpy as np
import pandas as pd
import csv
import os

labels = ["bed", "fall", "pickup", "run", "sitdown", "standup", "walk"]

# window length to perform short-term fourier transform
window_length = 1000
# threshold to check if we have certain amount of activities
threshold = 0.6
step = 200
downsample = 2

def preprocess_raw_csi_data(folder, save):
    """
    takes raw CSI data as input
    :return: returns numpoy array after performing fast fourier transform
    """
    ans = []
    for label in labels:
        label = label.lower()
        data_path = os.path.join(folder, "input_*" + label + "*.csv")
        input_data_files = glob.glob(data_path)
        # we sort the files to match file names of input and annotation files
        input_data_files = sorted(input_data_files)
        annotation_data_files = [os.path.basename(file).replace("input_","annotation_") for file in input_data_files]
        annotation_data_files = [os.path.join(folder,file_name) for file_name in annotation_data_files]
        feature = []
        index = 0
        for csi_file, annot_label_file in zip(input_data_files,annotation_data_files):
            index+=1
            if not os.path.exists(annot_label_file):
                print("label file doesn't exist")
                continue
            activity_data = []
            with open(annot_label_file,'r') as annot_file:
                reader = csv.reader(annot_file)
                for line in reader:
                    lbl = line[0]
                    if lbl == 'NoActivity':
                        activity_data.append(0)
                    else:
                        activity_data.append(1)
            activity = np.array(activity_data)
            csi = []
            with open(csi_file, 'r') as csi_data_file:
                reader = csv.reader(csi_data_file)
                for line in reader:
                    line_array = np.array([float(v) for v in line])
                    # we are extracting amplitude and we are removing phase
                    line_array = line_array[1:91]
                    csi.append(line_array[np.newaxis,...])
            csi = np.concatenate(csi, axis=0)
            ind = 0
            feature_target = []
            while ind + window_length <= csi.shape[0]:
                cur_activity = activity[ind:ind+window_length]
                if np.sum(cur_activity)  <  threshold * window_length:
                    ind += step
                    continue
                cur_feature = np.zeros((1, window_length, 90))
                cur_feature[0] = csi[index:index+window_length, :]
                feature_target.append(cur_feature)
                ind += step
            feature.append(np.concatenate(feature_target, axis=0))
            print('Finished {:.2f}% for Label {}'.format(index / len(input_data_files) * 100, label))

        feature_array = np.concatenate(feature, axis=0)
        if save:
            np.savez_compressed("X_{}_win_{}_thrshd_{}percent_step_{}.npz".format(
            label, window_length, int(threshold*100), step), feature_array)
        feat_label = np.zeros((feature_array.shape[0], len(labels)))
        feat_label[:, labels.index(label)] = 1
        ans.append(feature_array)
        ans.append(feat_label)
    numpy_tuple = tuple(ans)
    if downsample > 1:
        return tuple([v[:, ::downsample,...] if i%2 ==0 else v for i, v in enumerate(numpy_tuple)])
    return numpy_tuple



def load_csi_data_from_numpy_data(np_files):

    numpy_list = []
    if len(np_files) != 7:
        print("there should be seven labels data")
    else:
        x = [np.load(f)['arr_0'] for f in np_files]
        if downsample > 1:
            x = [arr[:,::downsample, :] for arr in x]
        y = [np.zeros((arr.shape[0], len(labels))) for arr in x]
        numpy_list = []
        for i in range(len(labels)):
            y[i][:,i] = 1
            numpy_list.append(x[i])
            numpy_list.append(y[i])
    return tuple(numpy_list)
