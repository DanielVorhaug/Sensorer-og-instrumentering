import pylab
import csv
import matplotlib.pyplot as plt
import numpy as np

header = []
data = []


filename = "Lab_1-System\Supply\Filter_network_better.csv"
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
ch3 = ch1+ch2
ch3 = ch3-ch3[0] # normaliser

fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.plot( time, ch3, "-", 16.75, -3, "ro", [0,16.75, 16.75], [-3,-3,-50], "-r")
ax1.grid()
ax1.set_xscale("log")
ax1.set_ylabel("Magnitude [dB]")
ax1.set_xlabel("Frekvens [Hz]")



pylab.show()
