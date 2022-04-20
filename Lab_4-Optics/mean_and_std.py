import pylab
from re import T
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import math


def getData(filepath, color = 0): #Gets the data of color with given index from file with the given filepath
    f = open(filepath)

    sat = []
    for line in f: #Takes out the value of the specific color we want
        values = line.split()
        sat.append(values[color])

    sat = np.ndarray.astype(np.array(sat), np.float32) #Makes the array into a numpy array of ints

    f.close()

    return(sat)

def find_pulse_autocorrelation(Ts, filepath): #Finds the pulse using autocorrelation (duh)
    pulse = []
    #max_bpm = 300
    max_hz = 5 #max_hz = max_bpm/60
    min_delay = 1/max_hz
    min_lags = math.floor(min_delay/Ts)

    for i in range(3): #Goes through and does it for all colors
        color_values = getData(filepath, i) #Gets data for the color
        color_values = signal.detrend(color_values)

        auto_corr = np.correlate(color_values, color_values, "full") #Calculates the autocorrelation
        auto_corr = auto_corr[auto_corr.shape[0]//2 + min_lags + 1:] #Removes negative delays and delays that would result in a too high frequency
        lags = np.argmax(auto_corr) + min_lags#Gives the delay as a number of lags from previous center (which is lag = 0)
        time_delay = Ts*lags
        pulse.append(60/time_delay) #60 * freq[hz] = freq[bpm]

    return(pulse)

def find_pulse(Ts,freq, spectrum):
    limit_hz = 40/60 #40 bpm omregnet til hz
    limit_sample = math.ceil(limit_hz * spectrum.shape[0] * Ts)

    peak = limit_sample + np.argmax(np.abs(spectrum[limit_sample:spectrum.shape[0]//2]))
    #peak = np.argmax(np.abs(spectrum))
    fd = abs(freq[peak]) * 60 #find the pulse and convert from Hz to bpm

    return(fd)

def calc_SNR_peaks(Ts, spectrum):
    upper_limit_hz = 225/60 #225 bpm omregnet til hz. 225 ble valgt siden det er en dødelig høy puls
    upper_limit_sample = math.ceil(upper_limit_hz * spectrum.shape[0] * Ts)
    lower_limit_hz = 40/60 #40 bpm omregnet til hz
    lower_limit_sample = math.ceil(lower_limit_hz * spectrum.shape[0] * Ts)
    
    spectrum = spectrum[lower_limit_sample:upper_limit_sample]
    sig_peak_index = np.argmax(np.abs(spectrum)) #Finner indexen til signaltopp for frekvenser over limit_hz

    signal_width = 10
    spectrum_sig_peak_removed = np.delete(spectrum, np.arange(sig_peak_index-signal_width, sig_peak_index+signal_width))
    noise_peak = np.max(np.abs(spectrum_sig_peak_removed)) #Finner verdien til støytoppen
    
    SNR = 20*np.log10(np.abs(spectrum[sig_peak_index]/noise_peak))
    return(SNR)

def calc_SNR_average(Ts, spectrum):
    upper_limit_hz = 225/60 #225 bpm omregnet til hz. 225 ble valgt siden det er en dødelig høy puls
    upper_limit_sample = math.ceil(upper_limit_hz * spectrum.shape[0] * Ts)
    lower_limit_hz = 40/60 #40 bpm omregnet til hz
    lower_limit_sample = math.ceil(lower_limit_hz * spectrum.shape[0] * Ts)

    spectrum = spectrum[lower_limit_sample:upper_limit_sample]
    
    sig_peak_index = np.argmax(np.abs(spectrum)) #Finner indexen til signaltopp for frekvenser over limit_hz

    signal_width = 10
    spectrum_sig_peak_removed = np.delete(spectrum, np.arange(sig_peak_index-signal_width, sig_peak_index+signal_width))        
    
    SNR = 20*np.log10(np.abs(spectrum[sig_peak_index]/np.average(spectrum_sig_peak_removed)))
    return(SNR)

filter_parameteres = [71,4,4] #window_length, polyorder and derivorder

color = ["  red", "green", " blue"]

pulses_fft = [[], [], []]
pulses_autocorr = [[], [], []]
SNRs_average = [[], [], []]
SNRs_peak = [[], [], []]

starting_index = 35
ending_index = 40

reference_bpm = 60

reference_array = np.full( ending_index-starting_index+1, reference_bpm)




for j in range(ending_index - starting_index + 1):
    for i in range(3):
        sample_frequency = 40 # [Hz]
        sample_period = 1 / sample_frequency # [s]
        
        index_string = str(j+starting_index)
        if len(index_string) == 1:
            index_string= "0" + index_string

        data = getData("Optics/Tests/Test" + index_string + "/result.txt", i)

        # Trims test
        data = data[100:-100]

        # Generate time axis
        num_of_samples = data.shape[0]  # returns shape of matrix
        data = signal.detrend(data)
        data_filtered = signal.savgol_filter(data, filter_parameteres[0], filter_parameteres[1], filter_parameteres[2], mode="constant")
        t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)


        #Generate frequency axis and take FFT
        freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
        spectrum = np.fft.fft(data, axis=0)  # takes FFT of all channels
        spectrum_filtered = np.fft.fft(data_filtered, axis=0)  # takes FFT of all channels

        # Filter low frequencies
        limit_hz = 40/60
        limit_sample = math.ceil(limit_hz * spectrum.shape[0] * sample_period)     


        #Find pulse using autocorrelation
        max_hz = 3.75 #max_hz = max_bpm/60, max_bpm = 225
        min_delay = 1/max_hz #minimum delay would be the inverse of the highest frequency possible
        min_lags = math.floor(min_delay/sample_period)

        max_delay = 1/limit_hz
        max_lags = math.floor(max_delay/sample_period)
        
        
        auto_corr = np.correlate(data, data, "full") #Calculates the autocorrelation
        auto_corr_filtered = np.correlate(data_filtered, data_filtered, "full") #Calculates the autocorrelation
        
        t_autocorr = np.linspace(start=-num_of_samples*sample_period, stop=num_of_samples*sample_period, num=2*num_of_samples-1)

        auto_corr_trimmed = auto_corr_filtered[auto_corr_filtered.shape[0]//2 + min_lags + 1:auto_corr_filtered.shape[0]//2 + max_lags + 1] #Removes negative delays and delays that would result in a too high frequency
        lags = np.argmax(auto_corr_trimmed) + min_lags + 1#Gives the delay as a number of lags from previous center (which is lag = 0)
        time_delay = sample_period*lags

        pulses_fft[i].append(find_pulse(sample_period, freq, spectrum_filtered))
        pulses_autocorr[i].append(60/time_delay)
        SNRs_average[i].append(calc_SNR_average(sample_period, spectrum_filtered))
        SNRs_peak[i].append(calc_SNR_peaks(sample_period, spectrum_filtered))

        print(f"Recording {index_string}: Channel {color[i]}: \tpulse (FFT): {pulses_fft[i][-1]:.1f} \tpulse (autocorr): {pulses_autocorr[i][-1]:.1f} \t\tSNR (average noise): {SNRs_average[i][-1]:.1f}dB\tSNR (peak noise): {SNRs_peak[i][-1]:.1f}dB")
    print()
    
    

for i in range(3):
    print(f"Pulse (FFT): \t\tChannel {color[i]}: \tAverage: {np.average(pulses_fft[i]):.2f}bpm,\tStd: {np.std(pulses_fft[i]):.2f}bpm \tStd from reference: {np.sqrt(np.mean(np.abs(pulses_fft[i] - reference_array)**2.)):.2f}bpm")
    print(f"Pulse (autocorr): \tChannel {color[i]}: \tAverage: {np.average(pulses_autocorr[i]):.2f}bpm,\tStd: {np.std(pulses_autocorr[i]):.2f}bpm \tStd from reference: {np.sqrt(np.mean(np.abs(pulses_autocorr[i] - reference_array)**2.)):.2f}bpm")
    print()
print()
    

for i in range(3):
    print(f"SNR (average): \t\tChannel {color[i]}: \tAverage: {np.average(SNRs_average[i]):.2f}dB,\tStd: {np.std(SNRs_average[i]):.2f}dB")
print()

for i in range(3):
    print(f"SNR (peak): \t\tChannel {color[i]}: \tAverage: {np.average(SNRs_peak[i]):.2f}dB,\tStd: {np.std(SNRs_peak[i]):.2f}dB")
print()

