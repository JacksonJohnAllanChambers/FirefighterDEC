import Mapping
import fireClasses
import main

def citizens(map):
    for x in range (550):
        for y in range(100):
            if Mapping.get_citizens == 1:
                print("Citizen detected at ({x},{y}). Citizen evacuated\n")
                Mapping.set_citzen(x,y,0)

def firefighters(map):
    i = 0
    while i < len(main.Fire_fighters):
            if main.Fire_fighters[i].fireEscape == 1:
                print("1 escape route left. Firefighter at ({main.Fire_fighters[i].x},{main.Fire_fighters[i].y}). Firefighter rescued via helecopter\n")
                Mapping.set_citzen(main.Fire_fighters[i].x,main.Fire_fighters[i].y,0)