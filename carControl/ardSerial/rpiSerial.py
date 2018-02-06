# -*- coding: utf-8 -*-
"""
servo
160 - макс влево
125 - середина
90  - макс вправо

0-назад, 1- стоп, 2 - вперед

"""
import time
import serial 
import struct

ser = serial.Serial('COM3', 9600)
time.sleep(2)
print('open. change 1')
ser.write(struct.pack('>3B', 125, 128, 1))
time.sleep(0.5)
print(ser.readline())

time.sleep(2)

print('change 2')
ser.write(struct.pack('>3B', 160, 128, 2))
time.sleep(0.5)
print(ser.readline())

time.sleep(2)

print('change 3')
ser.write(struct.pack('>3B', 90, 128, 0))
time.sleep(0.5)
print(ser.readline())

time.sleep(2)

print('change 4')
ser.write(struct.pack('>3B', 125, 0, 1))
time.sleep(0.5)
print(ser.readline())

time.sleep(2)

ser.close()