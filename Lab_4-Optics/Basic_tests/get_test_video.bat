ssh -x pi@hostname.local "raspivid -v -o test.h264"
scp pi@hostname.local:/home/pi/test.h264 test.h264
