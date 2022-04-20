scp adc_sampler.c pi@hostname.local:/home/pi/adc_sampler/adc_sampler.c 
scp Makefile pi@hostname.local:/home/pi/adc_sampler/Makefile 
ssh pi@hostname.local "cd /home/pi/adc_sampler/ && sudo make read"
PAUSE