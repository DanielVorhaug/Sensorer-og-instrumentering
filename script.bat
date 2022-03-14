ssh -x pi@hostname.local "sudo /home/pi/adc_sampler/adc_sampler 31500"
scp pi@hostname.local:/home/pi/adcData.bin adcData.bin
py .\radar.py
