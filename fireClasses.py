import numpy as np
import Mapping

matrix = np.random.rand(100, 550)

class RefillStation:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self. y = y

class Zone:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Drone:
    def __init__(self, zone, number = 0, x = 0, y = 0):
        self.number = number
        self.x = x
        self.y = y
        self.zone = zone
        self.water = 0
        self.capacity = 10

    def firefight(self):
        fireSeverity = self.location.getFireSeverity()
        if(fireSeverity>= 4):
            self.spray(self, fireSeverity)

    def spray(self, fireSeverity):
        for i in range(3, fireSeverity): # not sure if number is right #  
            if self.water >= 0:
                self.water -= 1
                fireSeverity -= 1
                set.severity(self.location, fireSeverity)
            else:
                self.waterRefill()
                break

    def waterRefill(self):
        #x, y = waterRoute()
        self.move(self, x, y)

    def move(self, x, y):
        self.x, self.y = x, y

    def location(self):
        return self.x, self.y

class fireFighter:
    def __init__(self, x, y):
        self.x
        self.y

    def location(self):
        return self.x, self.y
    
    def fireEscape(self):

        ogX, ogY = self.location

        x = []
        y = []

        for i in range (0, 5):
            a = Mapping.get_fire(x[i], y[i])

            if a !=0:
                fires = fires + 1

        if fires == 7:
            escape = 1

        else:
            escape = 0

        return escape


    

