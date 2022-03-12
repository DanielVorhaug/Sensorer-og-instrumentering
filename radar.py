import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as signal
import math
from subprocess import call
from time import sleep


def raspi_import(path, channels=5):
    """
    Import data produced using adc_sampler.c.
    Returns sample period and ndarray with one column per channel.
    Sampled data for each channel, in dimensions NUM_SAMPLES x NUM_CHANNELS.
    """

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype=np.uint16)
        data = data.reshape((-1, channels))
    return sample_period, data

def plot_FFTs(to_plot, f):
    number_of_plots = to_plot.shape[1]
    for i in range(number_of_plots):
        plt.subplot(number_of_plots, 1, i+1)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power [dB]")
        plt.plot(f, to_plot[:, i], ".")
        plt.grid()
    
    plt.show()

    return

def calc_speed(fd):

    #Removed a factor of 10^8 from f0 and c as they get divided by each other
    f0 = 24.13*10 #Frequency of the radar
    c = 3 #Speed of light in free space (close enough for our use)

    v = (c*fd)/(2*f0)

    return(v)

def find_speed(path, I_channel = 0, Q_channel = 1):
    sample_period, data_raw = raspi_import(path)

    #If you need to see the raw data
    # plt.plot(np.linspace(-len(data_raw)//2 + 1, len(data_raw)//2, len(data_raw)), data_raw[:, 2], ".")
    # plt.show()

    #Just some spring cleaning
    data = data_raw[:, 0:2] #removes data from ADCs not in use
    data = signal.detrend(data, axis=0)  #removes DC component for each channel (should not matter as we have a filter)
    sample_period *= 1e-6  # change unit to micro seconds

    FFT = sp.fft.fft(data, axis=0) #Calculates the FFT of each channel
    FFT = sp.fft.fftshift(FFT)
    freqs = sp.fft.fftfreq(n = data.shape[0], d=sample_period)
    freqs = sp.fft.fftshift(freqs)

    plot_FFTs(FFT, freqs)

    #Finds the frequency caused by movement
    peak = np.argmax(np.abs(FFT[:, I_channel])) #The channel does not matter, but we must choose one
    peak_freq = freqs[peak]

    #Calculates the speed of movement from the frequency
    speed = calc_speed(peak_freq)

    return(speed)

    


filepath = 'adcData.bin' #Filepath to datafile
find_speed(filepath)