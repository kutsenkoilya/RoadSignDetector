# import the necessary packages
from __future__ import print_function
from GUIController import GUIController
from imutils.video import VideoStream
import time

 
# initialize the video stream and allow the camera sensor to warmup
print("[INFO] warming up camera...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
 
# start the app
pba = GUIController(vs)
pba.root.mainloop()