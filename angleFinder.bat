::This script does not update you on where in the process it currently is, nice for my use but may be trash in other settings
@ssh pi@hostname.local "sudo /home/pi/adc_sampler/adc_sampler 31250"
@scp pi@hostname.local:/home/pi/adcData.bin adcData.bin
@py .\acoustics.py