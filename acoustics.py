import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import math

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
    if(l12 - l13 - 2*l23 < 0):
        angl = angl + np.pi
    
    return(angl)

def find_lags(n12, n23, n13):
    maxLags = 6
    n12 = n12[len(n12)//2-maxLags: len(n12)//2+maxLags+1]
    n23 = n23[len(n23)//2-maxLags: len(n23)//2+maxLags+1]
    n13 = n13[len(n13)//2-maxLags: len(n13)//2+maxLags+1]

    # Finds index of maximum value and translates it into lag in samples
    #print(str(np.argmax(n12)) + " " + str(np.argmax(n23)) + " " + str(np.argmax(n13))) Things are currently working and as such this only causes
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



#Channel  2 is "upper left", channel 3 is "lower mid", channel 4 is "upper right"
# Import data from bin file
sample_period, data_raw = raspi_import('adcData.bin')
data = data_raw[:,:]

data = signal.detrend(data, axis=0)  # removes DC component for each channel
sample_period *= 1e-6  # change unit to micro seconds

#Interpolate
interpolation_factor = 1
data, sample_period = interpolate(sample_period, data, interpolation_factor)

#make crosscorelations
n12, n23, n13 = make_correlations(data, 2, 3, 4)

#Find number of lags
l12, l23, l13 = find_lags(n12, n23, n13)

#print(f"Lags: 1-2:{l12}, 2-3:{l23}, 1-3:{l13}") Usefull for testing, useless right now

#Calculate the angle
theta = get_angle(l12, l23, l13)

print(f"Angle: {theta}")


# Plots:
plt.subplot(3, 1, 1)
plt.plot(np.linspace(-len(n12)//2-1,  len(n12)//2, len(n12)), n12, ".")
plt.subplot(3, 1, 2)
plt.plot(np.linspace(-len(n12)//2-1, len(n12)//2, len(n12)), n23, ".")
plt.subplot(3, 1, 3)
plt.plot(np.linspace(-len(n12)//2-1, len(n12)//2, len(n12)), n13, ".")



plt.show()
