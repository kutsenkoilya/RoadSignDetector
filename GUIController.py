# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 23:53:11 2018

@author: Илья
"""

from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import imutils
import cv2

from GPIOController import GPIOController

class GUIController:
    def __init__(self, vs):
        self.vs = vs
        self.frame = None
        self.thread = None
        self.stopEvent = None
        self.root = tki.Tk()
        self.panel = None
        self.gpiocontroller = GPIOController()
        
        tki.Button(self.root, text="<",command=self.goLeftFull).grid(row = 2, column = 1)
        tki.Button(self.root, text="\\",command=self.goLeft).grid(row = 2, column = 2)
        tki.Button(self.root, text="/\\",command=self.goStraight).grid(row = 2, column = 3)
        tki.Button(self.root, text="/",command=self.goRight).grid(row = 2, column = 4)
        tki.Button(self.root, text=">",command=self.goRightFull).grid(row = 2, column = 5)
        
        tki.Button(self.root, text="R",command=self.goBack).grid(row = 3, column = 1)
        tki.Button(self.root, text="P",command=self.doStop).grid(row = 3, column = 2)
        tki.Button(self.root, text="10",command=self.goFwdTen).grid(row = 3, column = 3)
        tki.Button(self.root, text="25",command=self.goFwdTFive).grid(row = 3, column = 4)
        tki.Button(self.root, text="40",command=self.goFwdFty).grid(row = 4, column = 5)
        
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()
        self.root.wm_title("Remote Car GUI")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)
    def videoLoop(self):
        try:
            while not self.stopEvent.is_set():
                self.frame = self.vs.read()
                self.frame = imutils.resize(self.frame, width=300)
                image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)
                if self.panel is None:
                    self.panel = tki.Label(image=image)
                    self.panel.image = image
                    self.panel.grid(row = 1, columnspan=5)
                else:
                    self.panel.configure(image=image)
                    self.panel.image = image
        except RuntimeError:
            print("[INFO] caught a RuntimeError")
    
    def goLeftFull(self):
        self.gpiocontroller.servoTurn(1)
    def goLeft(self):
        self.gpiocontroller.servoTurn(2)
    def goStraight(self):
        self.gpiocontroller.servoTurn(3)
    def goRight(self):
        self.gpiocontroller.servoTurn(4)
    def goRightFull(self):
        self.gpiocontroller.servoTurn(5)
    
    def goBack(self):
        self.gpiocontroller.setSpeed(10)
        self.gpiocontroller.wheelsGoForward()
        
    def doStop(self):
        self.gpiocontroller.setSpeed(0)
        
    def goFwdTen(self):
        self.gpiocontroller.setSpeed(10)
        self.gpiocontroller.wheelsGoBackward()
        
    def goFwdTFive(self):
        self.gpiocontroller.setSpeed(25)
        self.gpiocontroller.wheelsGoBackward()
        
    def goFwdFty(self):
        self.gpiocontroller.setSpeed(40)
        self.gpiocontroller.wheelsGoBackward()
        
    def onClose(self):
        print("[INFO] closing...")
        cv2.destroyAllWindows()  
        self.stopEvent.set()
        self.vs.stop()
        self.root.quit()
        self.root.destroy()
        
            
            
            
            