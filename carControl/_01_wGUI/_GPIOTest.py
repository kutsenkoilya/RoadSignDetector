
import GPIOController
import time

controller = GPIOController.GPIOController()

controller.setSpeed(10)
controller.wheelsGoForward()
#колёсам нужно примерно 10 сек на прогрев!!!
time.sleep(10)


try:

    controller.servoTurn(1)
    time.sleep(0.1)
    """
    i = 1
    while i<6:
        print(i)
        controller.setSpeed(i*10)
        controller.servoTurn(i)
        i = i + 1
        time.sleep(5)
    """
        
    controller.stopAll()
    
    
except KeyboardInterrupt:
    controller.stopAll()
