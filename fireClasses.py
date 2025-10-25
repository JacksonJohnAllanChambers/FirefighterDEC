import Mapping

xMax = 550
yMax = 100
xMin = 0
yMin = 0

class RefillStation:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y

class Zone:
    def __init__(self, xMax, yMax, xMin, yMin, station):
        self.xMax = xMax
        self.yMax = yMax
        self.xMin = xMin
        self.yMax = yMin
        self.station = station

class Drone:
    def __init__(self, z, number = 0, x = 0, y = 0,):
        self.number = number
        self.x = x
        self.y = y
        self.zone = z
        self.water = 0
        self.capacity = 10

    def firefight(self, i):
        fireSeverity = self.location.getFireSeverity()
        if(fireSeverity>= 4):
            self.spray(self, fireSeverity, i)

    def spray(self, fireSeverity, i):

        if i == 0:
            max = 10

        if 0 < i <= 15:
            max = 7

        if 15 < i <= 25:
            max = 5

        if 25 < i <= 40:
            max = 2

        if 40 < i:
            max = 0


        for i in range(3, fireSeverity): # not sure if number is right #  
            j = 0
            if (self.water >= 0) and (j <= max):
                self.water -= 1
                fireSeverity -= 1
                set.severity(self.location, fireSeverity)
                j = j + 1
            else:
                self.waterRefill()
                break

    def waterRefill(self):
        x, y = Mapping.waterRoute()
        self.move(self, x, y)

    def move(self, x, y):
        self.x, self.y = x, y

    def location(self):
        return self.x, self.y
    
    def searchFire(self):
        i = 0
        x, y = self.location()

        if self.water == 0:
            self.waterRefill()
            return
    
        while xMax >= x >= xMin:
         
         while yMax >= y >= yMin:
            i = i +1

            if y <= yMax :
                y = y + 1

            else:
                y = y-1

            if x <= xMax:
                x = x + 1
            
            else:
                x = x -1 

            if Mapping.get_fire(x, y) >= 4 and Mapping.get_wing(x, y) >= 4:
                self.fireFight(i)
                self.move(x, y)
                
            else:
                self.searchFire()
    
class fireFighter:
    def __init__(self, x, y):
        self.x
        self.y

    def location(self):
        return self.x, self.y
    
    def fireEscape(self):

        ogX, ogY = self.location

        x = [ogX+1, ogX + 1, ogX, ogX -1, ogX -1, ogX -1, ogX, ogX + 1]
        y = [ogY, ogY + 1, ogY + 1, ogY + 1, ogY, ogY- 1, ogY -1, ogY -1]

        for i in range (0, 5):
            a = Mapping.get_fire(x[i], y[i])

            if a !=0:
                fires = fires + 1

        if fires == 7:
            escape = 1

        else:
            escape = 0

        return escape


    

