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
import time

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
        tki.Button(self.root, text="/\\",command=self.goStraight).grid(row = 2, column = 2)
        tki.Button(self.root, text="<",command=self.goLeftFull).grid(row = 3, column = 1)
        tki.Button(self.root, text="S",command=self.doStop).grid(row = 3, column = 2)
        tki.Button(self.root, text=">",command=self.goRightFull).grid(row = 3, column = 3)
        tki.Button(self.root, text="\\/",command=self.goBack).grid(row = 4, column = 2)
        
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
                    self.panel.grid(row = 1, columnspan=3)
                else:
                    self.panel.configure(image=image)
                    self.panel.image = image
        except RuntimeError:
            print("[INFO] caught a RuntimeError")
    
    def goLeftFull(self):
        self.gpiocontroller.TrunLeft()
    def goStraight(self):
        self.gpiocontroller.GoForward()
    def goRightFull(self):
        self.gpiocontroller.TurnRight()
    def goBack(self):
        self.gpiocontroller.GoBackward()
    def doStop(self):
        self.gpiocontroller.Stop()
    def onClose(self):
        print("[INFO] closing...")
        self.gpiocontroller.stopAll()
        cv2.destroyAllWindows()  
        self.stopEvent.set()
        self.vs.stop()
        self.root.quit()
        self.root.destroy()
        
            
            
            
            
