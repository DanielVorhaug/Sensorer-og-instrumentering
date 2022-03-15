import pylab
import csv
import matplotlib.pyplot as plt
import numpy as np

header = []
data = []


filename = "Ichannel.csv"
with open(filename) as csvfile:
    csvreader = csv.reader(csvfile)

    header = next(csvreader)

    for datapoint in csvreader:

        values = [float(value) for value in datapoint]
        data.append(values)

print(header)
print(data[0])
print(data[1])

time = [p[0] for p in data]
ch1 = np.array([p[1] for p in data])
ch2 = np.array([p[2] for p in data])

fig = plt.figure()

ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(time,ch2+ch1)
ax1.set_xscale("log")
ax1.set_ylabel("Amplitude [dB]")
ax1.set_xlabel("Frekvens [Hz]")

ax1 = fig.add_subplot(3, 1, 2)
ax1.plot(time,ch2)
ax1.set_xscale("log")
ax1.set_ylabel("Amplitude [dB]")
ax1.set_xlabel("Frekvens [Hz]")

ax2 = fig.add_subplot(3, 1, 3)
ax2.plot(time,ch1)
ax2.set_xscale("log")
ax2.set_ylabel("Amplitude [dB]")
ax2.set_xlabel("Frekvens [Hz]")

fig1 = plt.figure()

###############################################################################

header = []
data = []

filename = "Qchannel.csv"
with open(filename) as csvfile:
    csvreader = csv.reader(csvfile)

    header = next(csvreader)

    for datapoint in csvreader:

        values = [float(value) for value in datapoint]
        data.append(values)

print(header)
print(data[0])
print(data[1])

time = [p[0] for p in data]
ch1 = np.array([p[1] for p in data])
ch2 = np.array([p[2] for p in data])

ax3 = fig1.add_subplot(3, 1, 1)
ax3.plot(time,ch2+ch1)
ax3.set_xscale("log")
ax3.set_ylabel("Amplitude [dB]")
ax3.set_xlabel("Frekvens [Hz]")

ax3 = fig1.add_subplot(3, 1, 2)
ax3.plot(time,ch2)
ax3.set_xscale("log")
ax3.set_ylabel("Amplitude [dB]")
ax3.set_xlabel("Frekvens [Hz]")

ax4 = fig1.add_subplot(3, 1, 3)
ax4.plot(time,ch1)
ax4.set_xscale("log")
ax4.set_ylabel("Amplitude [dB]")
ax4.set_xlabel("Frekvens [Hz]")

pylab.show()

# plt.plot(time,ch2)#, time,ch2)
# plt.ylabel("Amplitude [dB]")
# plt.xlabel("Frekvens [10^x Hz]")
# plt.xscale("log")
# plt.show()

    
