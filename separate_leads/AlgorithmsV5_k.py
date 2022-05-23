import numpy as np
import scipy.signal
from ecgdetectors import Detectors
import scipy.stats
import neurokit2 as nk
import time

def high_frequency_noise_filter(data, max_loss_passband, min_loss_stopband, sampling_frequency=500):
    order, normal_cutoff = scipy.signal.buttord(20, 30, max_loss_passband, min_loss_stopband, fs=sampling_frequency)
    iir_b, iir_a = scipy.signal.butter(order, normal_cutoff, fs=sampling_frequency)
    filtered_data = scipy.signal.filtfilt(iir_b, iir_a, data)
    return filtered_data


def baseline_filter(data, max_loss_passband, min_loss_stopband, sampling_frequency=500):
    order, normal_cutoff = scipy.signal.buttord(0.5, 8, max_loss_passband, min_loss_stopband, fs=sampling_frequency)
    iir_b, iir_a = scipy.signal.butter(order, normal_cutoff, fs=sampling_frequency)
    filtered_data = scipy.signal.filtfilt(iir_b, iir_a, data)
    return filtered_data


def stationary_signal_check(data, num_leads, window_length):
    res = []
    for lead in range(1, num_leads + 1):
        window_matrix = np.lib.stride_tricks.sliding_window_view(data[lead], window_length)[::10]
        for window in window_matrix:
            if np.amax(window) == np.amin(window):
                res.append(1)
                break
        if len(res) != lead:
            res.append(0)
    return res


def heart_rate_check(data, num_leads, heart_rate_limits, sampling_frequency, length_recording=10):
    res = []
    for lead in range(1, num_leads + 1):
        beats = Detectors(sampling_frequency).pan_tompkins_detector(data[lead])
        if len(beats) > ((heart_rate_limits[1]*length_recording)/60) or len(beats) < ((heart_rate_limits[0]*length_recording)/60):
            res.append(1)
        else:
            res.append(0)
    return res


def signal_to_noise_ratio_check(data, num_leads, SNR_threshold, signal_freq_band, sampling_frequency=500):
    res = []
    for lead in range(1, num_leads + 1):
        f, pxx_den = scipy.signal.periodogram(data[lead], fs=sampling_frequency, scaling="spectrum")
        if sum(pxx_den):
            signal_power = sum(pxx_den[(signal_freq_band[0]*10):(signal_freq_band[1]*10)])
            SNR = signal_power / (sum(pxx_den) - signal_power)
        else:
            res.append(0)
            continue
        if SNR < SNR_threshold:
            res.append(1)
        else:
            res.append(0)
    return res

def processing(ECG, num_leads, temp_freq, SNR_threshold, signal_freq_band, window_length, 
               heart_rate_limits, max_loss_passband, min_loss_stopband, sampling_frequency, length_recording=10):
    second = time.time()
    resampled_ECG = []
    if temp_freq != sampling_frequency:
        for n in range(0, num_leads + 1):
            resampled_ECG.append(nk.signal_resample(ECG[n], sampling_rate=int(temp_freq), desired_sampling_rate=sampling_frequency, method="numpy"))
    else:
        resampled_ECG = ECG

    filt_ECG = [resampled_ECG[0]]
    for lead in range(1, num_leads + 1):
        filter_noise =  high_frequency_noise_filter(ECG[lead], max_loss_passband, min_loss_stopband, sampling_frequency)
        filter_baseline = baseline_filter(ECG[lead], max_loss_passband, min_loss_stopband, sampling_frequency)
        filt_ECG.append(filter_noise - filter_baseline)

    SQM = []  # Signal Quality Matrix
    SQM.append(stationary_signal_check(ECG, num_leads, window_length))
    SQM.append(heart_rate_check(filt_ECG, num_leads,  heart_rate_limits, sampling_frequency, length_recording))
    SQM.append(signal_to_noise_ratio_check(ECG, num_leads, SNR_threshold, signal_freq_band, sampling_frequency,))

    combination = list("" for i in range(0, num_leads))
    for lead in range(1, num_leads + 1):
        combination[lead - 1] = SQM[0][lead - 1] + SQM[1][lead - 1] + SQM[2][lead - 1]

    res = []
    for x in range(0, num_leads):
        if combination[x] >= 1:
            combination[x] = u"\u2716"
        else:
            combination[x] = u"\u2714"

    for y in range(0, 3):
        SQM_print = []
        for x in range(0, num_leads):
            if SQM[y][x] == 1:
                SQM_print.append(u"\u2716")
            else:
                SQM_print.append(u"\u2714")
        res.append(SQM_print)

    res.append(combination)
    print(time.time()-second)
    return res

