ssh -x pi@hostname.local "python /home/pi/camera/record_video.py /home/pi/camera/video"
scp pi@hostname.local:/home/pi/camera/video.h264 video.h264
scp pi@hostname.local:/home/pi/camera/video.mp4 video.mp4
