"""
Управление сервоприводом = PIN #8  = Pin 14 = верхний ряд 4ый слева
Управление мотором 1     = PIN #10 = Pin 15 = верхний ряд 5ый слева
Управление мотором 2     = PIN #12 = Pin 18 = верхний ряд 6ой слева
Управление скоростью     = PIN #11 = Pin 17 = нижний ряд 6ой слева
"""

import RPi.GPIO as GPIO
import GPIOConstants

class GPIOController:

    GPIO.setmode(GPIO.BOARD)     
    servoState = 7.5

    GPIO.setup(GPIOConstants.PIN_SERVO, GPIO.OUT) #Pin #8
    servoObj = GPIO.PWM(GPIOConstants.PIN_SERVO,50)

    GPIO.setup(GPIOConstants.PIN_MOTOR_SPD,GPIO.OUT) 
    speedObj = GPIO.PWM(GPIOConstants.PIN_MOTOR_SPD,0.1)      #задание начальной частоты

    #Настройка колёс
    GPIO.setup(GPIOConstants.PIN_MOTOR_ONE,GPIO.OUT) #Pin #10
    GPIO.setup(GPIOConstants.PIN_MOTOR_TWO,GPIO.OUT) #Pin #12
    
    def __init__(self):
        #Настройка сервы
        self.servoObj.start(self.servoState)
        self.servoObj.ChangeDutyCycle(self.servoState) #начальный поворот в состояние 90 градусов (рулевое колесо в исходном положении)
        #Настройка скорости колёс
        self.speedObj.start(0)  #заданик начальной duty
        
    def wheelsGoForward(self):
        #едем вперед
        GPIO.output(GPIOConstants.PIN_MOTOR_ONE,GPIO.HIGH)
        GPIO.output(GPIOConstants.PIN_MOTOR_TWO,GPIO.LOW)

    def wheelsGoBackward(self):
        #едем назад
        GPIO.output(GPIOConstants.PIN_MOTOR_ONE,GPIO.LOW)
        GPIO.output(GPIOConstants.PIN_MOTOR_TWO,GPIO.HIGH)
        
    def setSpeed(self,speed):
        #Настройка скорости колёс
        #примерные значение скорости
        #speed: 
        #       = 0 - стоим
        #       = 10 - едем медленно
        #       ...
        #       = 40 - скорость не отличима от максимальной
        #       = 40-100 - макс значения скорости        
        self.speedObj.ChangeDutyCycle(speed)   #частота
        self.speedObj.ChangeFrequency(speed)   #duty
        
    def servoTurn(self,state):
        #пусть пока будет 5 состояний
        #1 - максимально налево
        #2 - наполовину налево
        #3 - прямо
        #4 - наполовину направо
        #5 - наполовину направо
        
        if (state == 1):
            self.servoState = 8.8 #полный поврот налево
        elif (state == 2):
            self.servoState = 8.2
        elif (state == 3):
            if (self.servoState>7.5):
                self.servoState = 7.3 #возвращение в исходное положение из левого состояния
            elif (self.servoState<7.5):
                self.servoState = 7.7 #возвращение в исходное положение из правого состояния
        elif (state == 4):
            self.servoState = 6.8
        elif (state == 5):
            self.servoState = 6.2 #полный поворот направо
        
        self.servoObj.ChangeDutyCycle(self.servoState)
            
    def stopAll(self):
        self.servoObj.stop()
        self.speedObj.stop()
        GPIO.cleanup()
