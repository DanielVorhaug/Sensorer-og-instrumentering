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

def update(val):
    parameters = list()
    for i in wins:
        parameters.append(i.val)
    filter_parameteres = parameters
#    window_length = int(wins[0].val)
    update_figure(filter_parameteres)
   

def update_figure(filter_parameteres):
    data_filtered = signal.savgol_filter(data, filter_parameteres[0], filter_parameteres[1], filter_parameteres[2], mode="constant")
    data_filtered = data_filtered * (data[np.argmax(data)]-data[np.argmin(data)]) / (data_filtered[np.argmax(data_filtered)]-data_filtered[np.argmin(data_filtered)])
    subplots[0].clear()
    subplots[0].plot(t, data, "b", t, data_filtered, "r")
    subplots[0].legend(['Unfiltered', 'Filtered'])
    

    freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
    spectrum = np.fft.fft(data, axis=0)  # takes FFT of all channels
    spectrum_filtered = np.fft.fft(data_filtered, axis=0)  # takes FFT of all channels

    
    
    #Fixes frequency axis
    freq = np.fft.fftshift(freq) * 60
    spectrum = np.fft.fftshift(spectrum)
    spectrum_filtered = np.fft.fftshift(spectrum_filtered)

    spectrum_filtered = spectrum_filtered * spectrum[np.argmax(np.abs(spectrum_filtered))] / spectrum_filtered[np.argmax(np.abs(spectrum_filtered))]
    subplots[1].clear()
    subplots[1].set_xscale("log")
    subplots[1].plot(freq, np.abs(spectrum), "b", freq, np.abs(spectrum_filtered), "r")
    subplots[1].legend(['Unfiltered', 'Filtered'])

    auto_corr_filtered = np.correlate(data_filtered, data_filtered, "full") #Calculates the autocorrelation
    auto_corr_filtered = auto_corr_filtered*auto_corr[auto_corr.shape[0]//2]/auto_corr_filtered[auto_corr_filtered.shape[0]//2]
    subplots[2].clear()
    subplots[2].plot(t_autocorr, auto_corr, "b", t_autocorr, auto_corr_filtered, "r")
    subplots[2].set_ylabel("Amplitude")
    subplots[2].set_xlabel("Autokorrelasjon [s]")
    subplots[2].legend(['Unfiltered', 'Filtered'])  

    plots[0].canvas.draw() #redraw the figure

plots = list()
subplots = list()
sliders = list()
wins = list()

t = []
t_autocorr = []
data = []
data_filtered = []
auto_corr = []
auto_corr_filtered = []



for i in range(1):
    i=1
    plots.append(plt.figure())
    sample_frequency = 40 # [Hz]
    sample_period = 1 / sample_frequency # [s]

    data = getData("Optics/Tests/Test5/result.txt", i)

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
    
    
    #Find pulse using autocorrelation
    max_hz = 5 #max_hz = max_bpm/60
    min_delay = 1/max_hz
    min_lags = math.floor(min_delay/sample_period)
    
    auto_corr = np.correlate(data, data, "full") #Calculates the autocorrelation
    auto_corr_filtered = np.correlate(data_filtered, data_filtered, "full") #Calculates the autocorrelation
    
    t_autocorr = np.linspace(start=-num_of_samples*sample_period, stop=num_of_samples*sample_period, num=2*num_of_samples-1)

    auto_corr_trimmed = auto_corr_filtered[auto_corr_filtered.shape[0]//2 + min_lags:] #Removes negative delays and delays that would result in a too high frequency
    lags = np.argmax(auto_corr_trimmed) + min_lags #Gives the delay as a number of lags from previous center (which is lag = 0)
    time_delay = sample_period*lags

    print(f"Channel {i} says the pulse is: {(60/time_delay):.3f}bpm using autocorrelation")

    
    # Normalize
    data_filtered = data_filtered * (data[np.argmax(data)]-data[np.argmin(data)]) / (data_filtered[np.argmax(data_filtered)]-data_filtered[np.argmin(data_filtered)])
    spectrum_filtered = spectrum_filtered * spectrum[np.argmax(np.abs(spectrum_filtered))] / spectrum_filtered[np.argmax(np.abs(spectrum_filtered))]
    auto_corr_filtered = auto_corr_filtered*auto_corr[auto_corr.shape[0]//2]/auto_corr_filtered[auto_corr_filtered.shape[0]//2]

    subplots.append(plots[-1].add_subplot(3, 1, 1))
    subplots[-1].plot(t, data, "b", t, data_filtered, "r")
    #subplots[-1].set_xscale("log")
    subplots[-1].set_ylabel("Amplitude")
    subplots[-1].set_xlabel("Frekvens [Hz]")
    subplots[-1].legend(['Unfiltered', 'Filtered'])


    subplots.append(plots[-1].add_subplot(3, 1, 2))
    subplots[-1].plot(freq, np.abs(spectrum), "b", freq, np.abs(spectrum_filtered), "r")
    subplots[-1].set_xscale("log")
    subplots[-1].set_ylabel("Amplitude")
    subplots[-1].set_xlabel("Puls [bpm]")
    subplots[-1].legend(['Unfiltered', 'Filtered'])
    subplots[-1].grid()

    subplots.append(plots[-1].add_subplot(3, 1, 3))
    subplots[-1].plot(t_autocorr, auto_corr, "b", t_autocorr, auto_corr_filtered, "r")
    subplots[-1].set_ylabel("Amplitude")
    subplots[-1].set_xlabel("Autokorrelasjon [s]")
    subplots[-1].legend(['Unfiltered', 'Filtered'])    

    plt.subplots_adjust(bottom=0.25)

    sliders.append(plt.axes([0.25, 0.1, 0.65, 0.03])) #xposition, yposition, width and height
    wins.append(Slider(sliders[-1], 'Window size', valmin=1, valmax=199, valinit=71, valstep=2))

    sliders.append(plt.axes([0.25, 0.15, 0.65, 0.03])) #xposition, yposition, width and height
    wins.append(Slider(sliders[-1], 'polyorder', valmin=0, valmax=11, valinit=4, valstep=1))

    sliders.append(plt.axes([0.25, 0.20, 0.65, 0.03])) #xposition, yposition, width and height
    wins.append(Slider(sliders[-1], 'derivorder', valmin=0, valmax=11, valinit=4, valstep=1))
    
    
    


    ##########################################################################################

    
for i in wins:
    i.on_changed(update)

plt.show()
pylab.show()
