import matplotlib
from os import listdir
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import butter, freqz, filtfilt, argrelextrema
from data_extract_ios import pull_data

matplotlib.rcParams['figure.figsize'] = (10, 5) 

def count_disturbances(filename, x, y, z, signal, timestamps):
    # Making timestamps start at 0
    c = timestamps[0]
    timestamps = (timestamps - c)

    # sample_rate = 1 / timestamps[1] == 99.8
    sample_rate = 100 # Hz
    
    # filter out the part of the signal that is taking phone in and out of the pocket
    samples_cut = sample_rate * 8   # 1st 8 seconds and last 8 seconds get cut 
    x = x[samples_cut: -samples_cut]
    y = y[samples_cut: -samples_cut]
    z = z[samples_cut: -samples_cut]
    timestamps = timestamps[samples_cut: -samples_cut]
    signal = signal[samples_cut: -samples_cut]


    # Filter requirements.
    order = 3
    cutoff = 2 # desired cutoff frequency of the filter, Hz

    # Create the filter.
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False) # low pass only filter

    filtered_signal = filtfilt(b, a, signal)

    # Finding peaks and filtering
    index_of_peaks = argrelextrema(filtered_signal, np.greater)[0] # returns array of all of the indexs of peaks/local maxima in the signal
    filtered_index_of_peaks = np.array([])
    for i in index_of_peaks:
        peak = filtered_signal[i]
        if peak > 4: #Filters out lower magnitude peaks that would not be steps -> this number can be modified
            filtered_index_of_peaks = np.append(filtered_index_of_peaks, i)

    # Peak values for graph and filteres peaks that are too close together
    disturbance_max_time = 5    # seconds 
    peaks = np.array([])
    peak_times = np.array([])
    for i in range(len(filtered_index_of_peaks)):
        # Filters out peaks that are too close together
        if i > 0 and filtered_index_of_peaks[i] - filtered_index_of_peaks[i-1] < (sample_rate * disturbance_max_time):
            continue
        peak_index = int(filtered_index_of_peaks[i])
        peaks = np.append(peaks, filtered_signal[peak_index])
        peak_times = np.append(peak_times, timestamps[peak_index])

    disturbances_counted = len(peaks)

    plt.figure()
    plt.title(f'Disturbances counted in {filename}')
    plt.plot(timestamps, filtered_signal, "-r", label='filtered signal')
    plt.plot(peak_times, peaks, "b*", label=f'disturbance counted ({disturbances_counted})')
    plt.legend(loc = 'upper center')
    plt.show()

for filename in listdir('data'):
    if filename == '.DS_Store':
        continue
    x, y, z, signal, timestamps = pull_data('data', filename)
    count_disturbances(filename, x, y, z, signal, timestamps)

