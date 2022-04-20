ssh -x pi@hostname.local "raspistill -v -o test.jpg"
scp pi@hostname.local:/home/pi/test.jpg test.jpg
