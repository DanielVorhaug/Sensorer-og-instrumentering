ssh -x pi@hostname.local "sudo /home/pi/adc_sampler/adc_sampler 10000"
scp pi@hostname.local:/home/pi/adcData.bin adcData.bin
py .\raspi_analyze.py
