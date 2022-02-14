ssh pi@hostname.local "sudo /home/pi/adc_sampler/adc_sampler 31250"
scp pi@hostname.local:/home/pi/adcData.bin adcData.bin
py .\raspi_analyze.py