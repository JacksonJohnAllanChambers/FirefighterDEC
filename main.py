from fireClasses import RefillStation, Drone
import Rescue, FirePrediction, Mapping, subprocess, time

sleepTime = 1

Yarmouth = RefillStation("Yarmouth", 0, 40),
Lunenburg = RefillStation("Lunenburg", 140, 25),
Windsor = RefillStation("Windsor", 250, 70),
Halifax = RefillStation("Halifax", 265, 30),
NewGlasgow = RefillStation("New Glasgow", 400, 75),
Sydeney = RefillStation("Sydney", 540, 50)

zoneOne = Zone(90, 110, 0, 0)
zoneTwo = Zone(180, 110, 91, 0)
zoneThree = Zone(260, 110, 181, 0)
zoneFour = Zone(370, 110, 261, 0)
zoneFive = Zone(460, 110, 371, 0)
zoneSix = Zone(550, 110, 461, 0)

droneOne = Drone(1, 0, 0)
droneTwo = Drone(2, 0, 0)
droneThree = Drone(3, 0, 0)
droneFour = Drone(4, 0, 0)
droneFive = Drone(5, 0, 0)
droneSix = Drone(6, 0, 0)

drones = [droneOne, droneTwo, droneThree, droneFour, droneFive, droneSix]

Real_map = [['\0' for _ in range(550)] for _ in range(100)]
Current_map = [['\0' for _ in range(550)] for _ in range(100)]
Predicted_map = [['\0' for _ in range(550)] for _ in range(100)]
Fire_fighters = []

def main():

    while(1):
        # get info, also handles rescuing citicens
        Mapping.scan_map(Current_map)

        #check if firefigter needs to escape
        Rescue.firefighters(Current_map)
    
        Predicted_map = FirePrediction.fire_prediction(Current_map).copy()

        for i in range(5):
            drones[i].Search() #this handles all the drones moves

        subprocess.run("-o firegen.exe")
        time.sleep(sleepTime)
    
    
    

    




    
    
