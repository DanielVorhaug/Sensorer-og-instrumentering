from re import T
from turtle import shape
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import math


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

def interpolate(sample_period, data, factor):
    new_data = np.zeros((data.shape[0] * factor, data.shape[1]))

    for i in range(data.shape[1]):
        num_of_samples = data.shape[0]
        old_t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)
        new_t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples * factor)
        new_data[:,i] = np.interp(x=new_t, xp=old_t, fp=data[:,i])
    
    new_sample_period = sample_period / factor
    return new_data, new_sample_period

# Import data from bin file
sample_period, data_raw = raspi_import('adcData.bin')
data = data_raw[5:,:]

data = signal.detrend(data, axis=0)  # removes DC component for each channel
sample_period *= 1e-6  # change unit to micro seconds

#Interpolate
interpolation_factor = 4
data, sample_period = interpolate(sample_period, data, interpolation_factor)

# Generate time axis
num_of_samples = data.shape[0]  # returns shape of matrix
t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)

# Generate frequency axis and take FFT
freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
spectrum = np.fft.fft(data, axis=0)  # takes FFT of all channels

#Fixes frequency axis from -16kHz to 16kHz
freq = np.fft.fftshift(freq)
spectrum = np.fft.fftshift(spectrum)


# Plot the results in two subplots
# NOTICE: This lazily plots the entire matrixes. All the channels will be put into the same plots.
# If you want a single channel, use data[:,n] to get channel n

for i in range(1,6):
    plt.subplot(5, 1, i)
    #plt.title("Time domain signal")
    plt.xlabel("Time [us]")
    plt.ylabel("Voltage")
    plt.plot(t[1:], data[1:,i-1], ".")

#for i in range(3,6):
#    plt.subplot(3, 1, i-2)
#    #plt.title("Power spectrum of signal")
#    plt.xlabel("Frequency [Hz]")
#    plt.ylabel("Power [dB]")
#    plt.plot(freq, 20*np.log10(np.abs(spectrum[:,i-1])))#, ".") # get the power spectrum

# print(freq[: math.ceil(len(freq)/2)])
# plt.subplot(2, 1, 1)
# #plt.title("Time domain signal")
# plt.xlabel("Time [us]")
# plt.ylabel("Voltage")
# plt.plot(t, data)

# plt.subplot(2, 1, 2)
# #plt.title("Power spectrum of signal")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Power [dB]")
# plt.plot(freq, 20*np.log10(np.abs(spectrum))) # get the power spectrum



plt.show()
