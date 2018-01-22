"""
Управление сервоприводом = PIN #8  = Pin 14 = верхний ряд 4ый слева
Управление мотором 1     = PIN #10 = Pin 15 = верхний ряд 5ый слева
Управление мотором 2     = PIN #12 = Pin 18 = верхний ряд 6ой слева
"""

import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)

GPIO.setup(8, GPIO.OUT) #Pin #8
p = GPIO.PWM(8,50)
p.start(7.5)
p.ChangeDutyCycle(7.5) #начальный поворот в состояние 90 градусов (рулевое колесо в исходном положении)

GPIO.setup(10,GPIO.OUT) #Pin #10
GPIO.setup(12,GPIO.OUT) #Pin #12

#едем вперед
GPIO.output(10,GPIO.HIGH)
GPIO.output(12,GPIO.LOW)S

#едем назад
#GPIO.output(10,GPIO.LOW)
#GPIO.output(12,GPIO.HIGH)

try:
    while True:
        #p.ChangeDutyCycle(2.5)  #сервоповорот в положение 0 градусов
        #p.ChangeDutyCycle(7.5)  #сервоповорот в положение 90 градусов
        #p.ChangeDutyCycle(12.5) #сервоповорот в положение 180 градусов
        #time.sleep(1) # ждём 1 секунду

        p.ChangeDutyCycle(6.2) #полный поворот направо
        time.sleep(1)
        p.ChangeDutyCycle(7.7) #возвращение в исходное положение из правого состояния
        time.sleep(1)
        p.ChangeDutyCycle(8.8) #полный поврот налево
        time.sleep(1)
        p.ChangeDutyCycle(7.3) #возвращение в исходное положение из левого состояния
        time.sleep(1)

except KeyboardInterrupt:
    p.stop()
    GPIO.cleanup()


