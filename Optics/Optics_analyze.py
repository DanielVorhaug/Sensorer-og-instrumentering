from random import sample
import pylab
from re import T
from turtle import shape
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
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
    limit_hz = 30/60 
    limit_sample = math.ceil(limit_hz * spectrum.shape[0] * Ts)

    peak = limit_sample + np.argmax(np.abs(spectrum[limit_sample:-limit_sample]))
    #peak = np.argmax(np.abs(spectrum))
    fd = freq[peak]

    return(fd)

filter_parameteres = [71,4,4] #window_length, polyorder and derivorder

plots = list()
subplots = list()

for i in range(3):
    plots.append(plt.figure())
    sample_frequency = 40 # [Hz]
    sample_period = 1 / sample_frequency # [s]

    data = getData("Optics/Tests/Test6/result.txt", i)

    # Trims test
    data = data[40:]

    # Generate time axis
    num_of_samples = data.shape[0]  # returns shape of matrix
    data = signal.detrend(data)
    data_filtered = signal.savgol_filter(data, filter_parameteres[0], filter_parameteres[1], filter_parameteres[2], mode="constant")
    t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)

    # Generate frequency axis and take FFT
    freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
    spectrum = np.fft.fft(data, axis=0)  # takes FFT of all channels
    spectrum_filtered = np.fft.fft(data_filtered, axis=0)  # takes FFT of all channels

    # Filter low frequencies
    limit_hz = 30/60
    limit_sample = math.ceil(limit_hz * spectrum.shape[0] * sample_period)
    # spectrum[-limit_sample:] = 0
    # spectrum[:limit_sample] = 0

    # spectrum_filtered[-limit_sample:] = 0
    # spectrum_filtered[:limit_sample] = 0

    print(f"Channel {i} says the pulse is: {find_pulse(sample_period, freq, spectrum)*60:.3f}bpm")

    #Fixes frequency axis
    freq = np.fft.fftshift(freq) * 60
    spectrum = np.fft.fftshift(spectrum)
    spectrum_filtered = np.fft.fftshift(spectrum_filtered)
    
    
    data_filtered = data_filtered * (data[np.argmax(data)]-data[np.argmin(data)]) / (data_filtered[np.argmax(data_filtered)]-data_filtered[np.argmin(data_filtered)])
    spectrum_filtered = spectrum_filtered * spectrum[np.argmax(np.abs(spectrum_filtered))] / spectrum_filtered[np.argmax(np.abs(spectrum_filtered))]
    


    subplots.append(plots[-1].add_subplot(2, 1, 1))
    subplots[-1].plot(t, data, "b", t, data_filtered, "r")
    #subplots[-1].set_xscale("log")
    subplots[-1].set_ylabel("Amplitude")
    subplots[-1].set_xlabel("Frekvens [Hz]")
    subplots[-1].legend(['Unfiltered', 'Filtered'])


    subplots.append(plots[-1].add_subplot(2, 1, 2))
    subplots[-1].plot(freq, np.abs(spectrum), "b", freq, np.abs(spectrum_filtered), "r")
    subplots[-1].set_xscale("log")
    subplots[-1].set_ylabel("Amplitude")
    subplots[-1].set_xlabel("Puls [bpm]")
    subplots[-1].legend(['Unfiltered', 'Filtered'])
    subplots[-1].grid()


    
    
    


    ##########################################################################################

    


plt.show()
pylab.show()
