import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import math

def make_correlations(data, wanted_axis1, wanted_axis2, wanted_axis3): #This function will currently also plot the autocorrelations
    # creates an array with crosscorrelations
    max_lag = 10  # The maximum lag possible with current configuration

    # Start with making the three full crosscorrelations
    # n12 = np.correlate(data[:,wanted_axis1], data[:,wanted_axis2], 'same')
    # n23 = np.correlate(data[:,wanted_axis2], data[:,wanted_axis3], 'same')
    # n13 = np.correlate(data[:,wanted_axis1], data[:,wanted_axis3], 'same')

    # autocorrelations
    n12 = np.correlate(data[:,wanted_axis1], data[:,wanted_axis1], 'same')
    n23 = np.correlate(data[:,wanted_axis2], data[:,wanted_axis2], 'same')
    n13 = np.correlate(data[:,wanted_axis3], data[:,wanted_axis3], 'same')

    plt.subplot(3, 1, 1)
    plt.plot(np.linspace(-len(n12)//2-1,  len(n12)//2, len(n12)), n12)
    plt.subplot(3, 1, 2)
    plt.plot(np.linspace(-len(n12)//2-1, len(n12)//2, len(n12)), n23)
    plt.subplot(3, 1, 3)
    plt.plot(np.linspace(-len(n12)//2-1, len(n12)//2, len(n12)), n13)

    # Then remove all values that would be for longer delays than possible
    # n12 = n12[len(n12) // 2 - max_lag: len(n12) // 2 + max_lag]
    # n23 = n23[len(n23) // 2 - max_lag: len(n23) // 2 + max_lag]
    # n13 = n13[len(n13) // 2 - max_lag: len(n13) // 2 + max_lag]

    return(n12, n23, n13)

def find_lags(n12, n23, n13):
    # Finds index of maximum value and translates it into lag in samples
    l12 = -(np.argmax(n12) - len(n12)//2)
    l23 = -(np.argmax(n23) - len(n23)//2)
    l13 = -(np.argmax(n13) - len(n13)//2)
    return(l12, l23, l13)

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


#Channel  2 is "upper left", channel 3 is "lower mid", channel 4 is "upper right"
# Import data from bin file
sample_period, data = raspi_import('Data\\09adcData Testing med 600Hz sinus.bin')

data = signal.detrend(data, axis=0)  # removes DC component for each channel
sample_period *= 1e-6  # change unit to micro seconds

#make crosscorelations
n12, n23, n13 = make_correlations(data, 2, 3, 4)

#Find number of lags
l12, l23, l13 = find_lags(n12, n23, n13)

# Generate time axis
num_of_samples = data.shape[0]  # returns shape of matrix
t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)

# Generate frequency axis and take FFT
freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
spectrum = np.fft.fft(data, axis=0)  # takes FFT of all channels

#Fixes frequency axis from -16kHz to 16kHz
freq = np.fft.fftshift(freq)
spectrum = np.fft.fftshift(spectrum)

print(l12, l23, l13)


# Plot the results in two subplots
# NOTICE: This lazily plots the entire matrixes. All the channels will be put into the same plots.
# If you want a single channel, use data[:,n] to get channel n

# for i in range(1,6):
#     plt.subplot(5, 1, i)
#     #plt.title("Time domain signal")
#     plt.xlabel("Time [us]")
#     plt.ylabel("Voltage")
#     plt.plot(t, data[:,i-1])

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
