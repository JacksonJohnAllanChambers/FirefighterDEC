from fireClasses import RefillStation, Drone
import Rescue, FirePrediction, Mapping

Yarmouth = RefillStation("Yarmouth", 0, 40),
Lunenburg = RefillStation("Lunenburg", 140, 25),
Windsor = RefillStation("Windsor", 250, 70),
Halifax = RefillStation("Halifax", 265, 30),
NewGlasgow = RefillStation("New Glasgow", 400, 75),
Sydeney = RefillStation("Sydney", 540, 50)

droneOne = Drone(1, 0, 0)
droneTwo = Drone(2, 0, 0)
droneThree = Drone(3, 0, 0)
droneFour = Drone(4, 0, 0)
droneFive = Drone(5, 0, 0)
droneSix = Drone(6, 0, 0)

drones = [droneOne, droneTwo, droneThree, droneFour, droneFive, droneSix]


Current_map = [['\0' for _ in range(550)] for _ in range(100)]
Predicted_map = [['\0' for _ in range(550)] for _ in range(100)]
Fire_fighters = []

def main():
    # get info, also handles rescuing citicens
    Mapping.scan_map(Current_map)

    #check if firefigter needs to escape
    Rescue.firefighters(Current_map)
   
    Predicted_map = FirePrediction.fire_prediction(Current_map).copy()

    for i in range(5):
        drones[i].Search() #this handles all the drones moves
    
    

    




    
    
