
import serial 
import struct
import time

class GPIOController:
    ser = serial.Serial('/dev/ttyUSB0', 9600)
    
    def __init__(self):
        time.sleep(0.5)
        print("ready")
        
    def GoForward(self):
        self.ser.write(struct.pack('>3B', 125, 128, 2)) 
	
    def GoBackward(self):
        self.ser.write(struct.pack('>3B', 125, 128, 0)) 

    def TrunLeft(self):
        self.ser.write(struct.pack('>3B', 160, 128, 2)) 
        
    def TurnRight(self):
        self.ser.write(struct.pack('>3B', 90, 128, 2)) 

    def Stop(self):
        self.ser.write(struct.pack('>3B', 125, 128, 1)) 
    
    def stopAll(self):
        self.ser.close()
