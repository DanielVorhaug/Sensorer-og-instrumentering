from cmath import log10
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as signal
import math
from subprocess import call
from time import sleep

test_distance = np.array([4.0])
test_time = np.array([ [4.48, 5.42, 4.44, 4.07, 5.54, 5.89], [3.33, 3.38, 3.42, 3.32, 0.0, 0.0], [2.18, 2.14, 2.00, 1.98, 0.0, 0.0], [2.49, 2.57, 2.61, 2.57, 0.0, 0.0], [1.74, 1.70, 1.78, 1.85, 0.0, 0.0], [1.06, 0.99, 1.20, 1.26, 0.0, 0.0]])
test_velocity = np.true_divide(test_time, test_distance)

print(test_time)
print(test_velocity)


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

def plot_raw(to_plot, Ts,  I_channel = 0):
    t=np.linspace(0, Ts*to_plot.shape[0], to_plot.shape[0]) #There is a better function than linspace for this, but I do not remeber the name

    number_of_plots = to_plot.shape[1]
    for i in range(number_of_plots):
        plt.subplot(number_of_plots, 1, i+1)
        if(i == I_channel):
            plt.title("I-channel")
        else:
            plt.title("Q-channel")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power [dB]")
        plt.plot(t, to_plot[:, i], ".")
        plt.grid()
    
    plt.show()

    return

def plot_FFTs(to_plot, Ts, I_channel = 0):
    FFT = sp.fft.fft(to_plot, axis=0) #Calculates the FFT of each channel
    freqs = sp.fft.fftfreq(to_plot.shape[0], Ts)

    #shifting frequencies for nicer plots
    FFT = sp.fft.fftshift(FFT)
    freqs = sp.fft.fftshift(freqs)

    #Calculates power spectrum in dB
    spectrum = 20*np.log10(FFT)

    number_of_plots = FFT.shape[1]
    for i in range(number_of_plots):
        plt.subplot(number_of_plots, 1, i+1)
        if(i == I_channel):
            plt.title("I-channel")
        else:
            plt.title("Q-channel")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power [dB]")
        plt.plot(freqs, spectrum[:, i], ".")
        plt.grid()
    
    plt.show()

    return

def plot_complex_FFT(to_plot, Ts, I_channel = 0, Q_channel = 1):
    complex_data = to_plot[:, I_channel] + to_plot[:, Q_channel]*1j
    FFT = sp.fft.fft(complex_data)
    freqs = sp.fft.fftfreq(complex_data.shape[0], Ts)

    #Frequency shift for better plot
    FFT = np.abs(sp.fft.fftshift(FFT))
    freqs = sp.fft.fftshift(freqs)

    #Calculates power spectrum in dB
    spectrum = 20*np.log10(FFT)

    plt.plot(freqs, spectrum, ".")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power [dB]")
    plt.title("Complex FFT")
    plt.grid()
    plt.show()

    return

def calc_SNR(sig, Ts, I_channel, Q_channel):
    complex_data = sig[:, I_channel] + sig[:, Q_channel]*1j
    FFT = sp.fft.fft(complex_data)

    #Finds the frequency caused by movement
    peak = np.argmax(np.abs(FFT))
    
    n = FFT.shape[0]

    noise = np.delete(FFT, [math.floor(peak-n*0.01), math.ceil(peak+n*0.01)] )

    noise_avg = np.average(np.abs(noise))

    SNR = 20*np.log10(np.abs(FFT[peak])/noise_avg)

    return(SNR)

def calc_speed(sig, Ts,  I_channel = 0, Q_channel = 1):
    #Removed a factor of 10^8 from f0 and c as they get divided by each other
    f0 = 24.13*10 #Frequency of the radar
    c = 3 #Speed of light in free space (close enough for our use)

    complex_data = sig[:, I_channel] + sig[:, Q_channel]*1j
    FFT = sp.fft.fft(complex_data)
    freqs = sp.fft.fftfreq(complex_data.shape[0], Ts)

    limit_hz = 60
    limit_sample = math.ceil(limit_hz * complex_data.shape[0] * Ts)
    
    #Finds the frequency caused by movement
    peak = limit_sample + np.argmax(np.abs(FFT[limit_sample:-limit_sample])) #The channel does not matter, but we must choose one
    fd = freqs[peak] #Doppler frequency (hopefully)

    v = (c*fd)/(2*f0)

    return(v)

def calc_var(groupNr, resultNr, I_channel=0, Q_channel=1):
    speeds = []
    for i in range(resultNr):
        f = "Radar_test_gjennomgang/" + str(groupNr) + str(i) + ".bin"
        sample_period, data_raw = raspi_import(f)
        data = data_raw[10:, 0:2] #removes data from ADCs not in use (Requires that I and Q use ADC 0 and 1)
        data = signal.detrend(data, axis=0)  #removes DC component for each channel (should not matter as we have a filter)
        sample_period *= 1e-6  # change unit to micro seconds

        speeds.append(calc_speed(data, sample_period, I_channel, Q_channel))

    variance = np.var(speeds)
    return(variance)

def plot_some_results(I_channel = 0, Q_channel = 1):
    for i in range(6): #6 is the number of different speeds we tested at
        f = "Radar_test_gjennomgang/" + str(i) + "0" + ".bin"
        sample_period, data_raw = raspi_import(f)
        data = data_raw[10:, 0:2] #removes data from ADCs not in use (Requires that I and Q use ADC 0 and 1)
        data = signal.detrend(data, axis=0)  #removes DC component for each channel (should not matter as we have a filter)
        sample_period *= 1e-6  # change unit to micro seconds

        plot_complex_FFT(data, sample_period, I_channel, Q_channel)

    return

def find_speed(path, I_channel = 0, Q_channel = 1):
    sample_period, data_raw = raspi_import(path)

    #If you need to see the raw data
    # plt.plot(np.linspace(-len(data_raw)//2 + 1, len(data_raw)//2, len(data_raw)), data_raw[:, 2], ".")
    # plt.show()

    #Just some spring cleaning
    data = data_raw[10:, 0:2] #removes data from ADCs not in use (Requires that I and Q use ADC 0 and 1)
    data = signal.detrend(data, axis=0)  #removes DC component for each channel (should not matter as we have a filter)
    sample_period *= 1e-6  # change unit to micro seconds

    #Calculates the speed of movement from the frequency
    speed = calc_speed(data, sample_period, I_channel, Q_channel)
    
    #plot_FFTs(data, sample_period, I_channel)
    plot_complex_FFT(data, sample_period, I_channel, Q_channel)

    #Calculates SNR
    #print(calc_SNR(data, sample_period, I_channel, Q_channel))

    return(speed)

    


filepath = 'adcData.bin' #Filepath to datafile
v = find_speed(filepath)
print("Velocity: " , v)
