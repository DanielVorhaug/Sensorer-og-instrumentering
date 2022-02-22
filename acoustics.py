import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import math
from subprocess import call
from time import sleep

def make_correlations(data, wanted_axis1, wanted_axis2, wanted_axis3):
    # Makes three full crosscorrelations
    n12 = np.correlate(data[:,wanted_axis1], data[:,wanted_axis2], 'same')
    n23 = np.correlate(data[:,wanted_axis2], data[:,wanted_axis3], 'same')
    n13 = np.correlate(data[:,wanted_axis1], data[:,wanted_axis3], 'same')

    # autocorrelations, was used for testing
    # n12 = np.correlate(data[:,wanted_axis1], data[:,wanted_axis1], 'same')
    # n23 = np.correlate(data[:,wanted_axis2], data[:,wanted_axis2], 'same')
    # n13 = np.correlate(data[:,wanted_axis3], data[:,wanted_axis3], 'same')

    return(n12, n23, n13)


def get_angle(l12, l23, l13):
    angl = np.arctan(np.sqrt(3))*(-l12 - l13)/(-l12 +l13 + 2*l23)
    #angl -= 1.05 #To control what is 0 radians
    if(l12 - l13 - 2*l23 < 0):
        angl = angl + np.pi
    
    return(angl)

def find_lags(n12, n23, n13, N_interp):
    maxLags = 10*N_interp
    n12 = n12[len(n12)//2-maxLags: len(n12)//2+maxLags+1]
    n23 = n23[len(n23)//2-maxLags: len(n23)//2+maxLags+1]
    n13 = n13[len(n13)//2-maxLags: len(n13)//2+maxLags+1]

    # Finds index of maximum value and translates it into lag in samples
    print(str(np.argmax(n12)-maxLags) + " " + str(np.argmax(n23)-maxLags) + " " + str(np.argmax(n13)-maxLags))
    l12 = -(np.argmax(n12) - maxLags)
    l23 = -(np.argmax(n23) - maxLags)
    l13 = -(np.argmax(n13) - maxLags)
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


def interpolate(sample_period, data, factor):
    new_data = np.zeros((data.shape[0] * factor, data.shape[1]))

    for i in range(data.shape[1]):
        num_of_samples = data.shape[0]
        old_t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)
        new_t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples * factor)
        new_data[:,i] = np.interp(x=new_t, xp=old_t, fp=data[:,i])
    
    new_sample_period = sample_period / factor
    return new_data, new_sample_period


def importAndCalculate(plotCorrs = 0):
    #Channel  2 is "upper left", channel 3 is "lower mid", channel 4 is "upper right"
    # Import data from bin file
    sample_period, data_raw = raspi_import('adcData.bin')
    data = data_raw[5:,:]

    data = signal.detrend(data, axis=0)  # removes DC component for each channel
    sample_period *= 1e-6  # change unit to micro seconds

    #Interpolate
    interpolation_factor = 3
    #data, sample_period = interpolate(sample_period, data, interpolation_factor) #Caused major issues so commented out for now

    #make crosscorelations
    n12, n23, n13 = make_correlations(data, 2, 3, 4)

    #Find number of lags
    l12, l23, l13 = find_lags(n12, n23, n13, interpolation_factor)

    #print(f"Lags: 1-2:{l12}, 2-3:{l23}, 1-3:{l13}") Usefull for testing, useless right now

    #Calculate the angle
    theta = get_angle(l12, l23, l13)

    print(f"Angle: {theta}")

    if(plotCorrs):
        plot_correlations(n12, n23, n13)

    return(theta)

def plot_correlations(n12, n23, n13):
    #Plots: commented out for quick testing
    plt.subplot(3, 1, 1)
    plt.plot(np.linspace(-len(n12)//2-1,  len(n12)//2, len(n12)), n12, ".")
    plt.subplot(3, 1, 2)
    plt.plot(np.linspace(-len(n12)//2-1, len(n12)//2, len(n12)), n23, ".")
    plt.subplot(3, 1, 3)
    plt.plot(np.linspace(-len(n12)//2-1, len(n12)//2, len(n12)), n13, ".")
    plt.show()

    return

def sampleAndStart(): #The function that calls for the RPi to begin sampling, transfer to local and then calls function importAndCalculate
    N_runs = int(input("How many times do you want to run the test?"))
    waitTime = 30
    angles = np.zeros(N_runs)

    for i in range(N_runs):
        sleep(waitTime)
        call(r'angleFinder.bat')
        angles[i] = importAndCalculate() #Starts importAndCalculate and stores the found angle for later use
        print(f"Test {i} finished, waiting {waitTime} seconds.")

    for ang in angles:
        print(ang, sep = " , ", end=" ")

    return()

sampleAndStart()