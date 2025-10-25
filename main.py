from fireClasses import RefillStation, Drone, Zone
import Rescue, FirePrediction, Mapping, subprocess, time

sleepTime = 1

Yarmouth = RefillStation("Yarmouth", 0, 40),
Lunenburg = RefillStation("Lunenburg", 140, 25),
Windsor = RefillStation("Windsor", 250, 70),
Halifax = RefillStation("Halifax", 265, 30),
NewGlasgow = RefillStation("New Glasgow", 400, 75),
Sydney = RefillStation("Sydney", 540, 50)

droneOne = Drone(1, 0, 0, 90, 110, 0, 0, Yarmouth)
droneTwo = Drone(2, 91, 0, 180, 110, 91, 0, Lunenburg)
droneThree = Drone(3, 181, 0, 260, 110, 181, 0, Windsor)
droneFour = Drone(4, 261, 0, 370, 110, 261, 0, Halifax)
droneFive = Drone(5, 371, 0, 460, 110, 371, 0, NewGlasgow)
droneSix = Drone(6, 461, 0, 550, 110, 461, 0, Sydney)

drones = [droneOne, droneTwo, droneThree, droneFour, droneFive, droneSix]

Real_map = [['\0' for _ in range(550)] for _ in range(100)]
Current_map = [['\0' for _ in range(550)] for _ in range(100)]
Predicted_map = [['\0' for _ in range(550)] for _ in range(100)]
Fire_fighters = []

round = -1

def main():

    while(1):
        new_round = open("your_file.txt").readline().strip()[0]
        if new_round == round:
            continue #wait for file change
        else:
            round = new_round
        

        # get info, also handles rescuing citicens
        Mapping.scan_map(Current_map)

        #check if firefigter needs to escape
        Rescue.firefighters(Current_map)
    
        Predicted_map = FirePrediction.fire_prediction(Current_map).copy()

        for i in range(5):
            drones[i].searchFire() #this handles all the drones moves

        Mapping.submit_moves(Current_map)

    
    
    

    




    
    
