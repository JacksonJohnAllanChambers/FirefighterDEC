#mapping
import sys
import math
import main
import Rescue
import fireClasses

import numpy as np
from typing import Union

# for string xxxyyyfwcf
# x cord (0-2)
# y cord (3-4)
# fire (6)
# severity(7)
# wind (8)
# direction (9)
# citizens (10)
# firefighters (11)
# turns since seen (12)
# trust value (13)


WIDTH, HEIGHT = 550, 100

Current_map = [['\0' for _ in range(550)] for _ in range(100)]
Predicted_map = [['\0' for _ in range(550)] for _ in range(100)]

##working main, will really be in other fike just showing how function
def main():
    #create initial map of null characters


    scan_map(Current_map)

    while(1):
        Current_map = Predicted_map #update predictions
        scan_map(Current_map)
         


def scan_map(map):

    #   drone#.location() -> returns xxxyy


    #scan data
    with open('map.txt', 'r') as file: # open file for scaning
        for line in file:
                #get data
                content = line.strip()
                xcord = content[0] + content[1] + content [2]
                ycord = content[3] + content[4]

                #Check if 
                for location in range(6):
                    #get drone location
                    drone_location = drone_location(location)
                    #check if in range of drone
                    if (drone_location.x -1 <= int(xcord) <= drone_location.x + 1) and (drone_location.y -1 <= int(ycord) <= drone_location.y + 1):
                        #get rest of data
                        fire = content[6]
                        wind = content[7]
                        direction = content[8]
                        Rescue.rescue_citizens(xcord,ycord)
                        citizens = 0
                        firefighter = content[10]
                        main.Fire_fighters.append(fireClasses.fireFighter(xcord,ycord))
                        map[xcord][ycord] = xcord + ycord + fire + wind + citizens + firefighter + '0' #zeros since it is seen by a drone
                    else:
                        map[xcord][ycord] = str(int(map[xcord][ycord]) + 1) # update turns since seen


def submit_moves(Submitted_map):
     with open('map.txt', 'w') as file:
          for x in range(550):
               for y in range(100):
                    file.write(Submitted_map[x][y] + '\n')

def update_predicted_map(map):
    main.Predicted_map = map

def get_info(x,y):
    return main.Current_map[x][y]
def set_info(x,y,val):
    main.Current_map[x][y] = val

def get_fire(x,y):
    return main.Current_map[x][y][6]
def set_fire(x,y,val):
    main.Current_map[x][y][6] = val

def get_sevarity(x,y):
    return main.Current_map[x][y][7]
def set_sevarity(x,y,val):
    main.Current_map[x][y][7] = val

def get_wind(x,y):
    return main.Current_map[x][y][8]
def set_wind(x,y,val):
    main.Current_map[x][y][8] = val

def get_direction(x,y):
    return main.Current_map[x][y][9]

def get_citizens(x,y):
    return main.Current_map[x][y][10]
def set_citizens(x,y,val):
    main.Current_map[x][y][10] = val

def get_firefighter(x,y):
    return main.Current_map[x][y][11]
def set_firefighter(x,y,val):
    main.Current_map[x][y][11] = val

def get_lastseen(x,y):
    return main.Current_map[x][y][12]  

def get_trust(x,y):
    return main.Current_map[x][y][13] 
def set_tust(x,y,val):
    main.Current_map[x][y][13] = val
     

##water route
def get_refil_path(Drone):
    #define water stations
    station = [
            [0,40],
            [140,25],
            [250,70],
            [265,30],
            [400,75],
            [540,50]

    ]

    x = Drone.x
    y = Drone.y

    #get closest refill station
    distance = math.sqrt( ((x-station[1][1])**2) + ((y-station[1][2])**2) ) #assume first is closest

    for i in range(7):
        temp = math.sqrt( ((x-station[i][1])**2) + ((y-station[i][2])**2) )
        if distance > temp: 
            distance = temp
            closest_station = i
        elif distance == temp:
            #idk yet
            temp == temp
    
    #path determine
    station_x = station[closest_station][1]
    station_y = station[closest_station][2]

    #determine favoured direction
    if abs(x - station_x) > abs(y - station_y):
        favoured_direction = 'x'
    else:
        favoured_direction = 'y'

    #detirmine up or down 1 = up, 0 = on the money -1 = down
    if x - station_x > 0:
        x_direction = -1
    elif x == station_x:
        x_direction = 0
    else:
        x_direction = 1
    
    #determine left or right -1 = left, 0 = on the money, 1 = right
    if y - station_y > 0:
        y_direction = -1
    elif y == station_y:
        y_direction = 0
    else:
        y_direction = 1
    
    #move loop
    moves = 50
    while( ((x != station_x) or (y != station_y)) and (moves > 0) ): #continue untill its at station
        moves -= 1
        match favoured_direction:
            case 'x':
                #check adjacent map values to see if known
                if 0 <= x + x_direction <= 550:
                    if  (get_trust(x + x_direction,y) >= 10 or get_trust(x + x_direction, y) == '\0') and (x != station_x): #check x
                        x += x_direction
                        continue
                if 0 <= y + y_direction <= 100:
                    if (get_trust(x, y + y_direction) > 10 or get_trust(x_direction, y + y_direction) == '\0') and (y != station_y):
                        y += y_direction
                        continue
                if x == station_x:
                    y += y_direction
                else:
                    x += x_direction


            case 'y':
                if 0 <= y + y_direction <= 100:
                    if (get_trust(x, y + y_direction) > 10 or get_trust(x_direction, y + y_direction) == '\0') and (y != station_y):
                        y += y_direction
                        continue
                if 0 <= x + x_direction <= 550:
                    if  (get_trust(x + x_direction,y) >= 10 or get_trust(x + x_direction, y) == '\0') and (x != station_x): #check x
                        x += x_direction
                        continue
                if x == station_x:
                    y += y_direction
                else:
                    x += x_direction

        Drone.move(x,y)

def in_bounds(x, y): 
    return 0 <= x < WIDTH and 0 <= y < HEIGHT

def trust_from_neighbors_tiles(grid, x: int, y: int):
    """
    grid[y][x] is either '\\0' or a tile string.
    n is the character at n_idx, T is the character at T_idx within the string.
    """
    n_idx: int = 12
    T_idx: int = 13
    if not in_bounds(x, y):
        return '\0'

    nbrs = []
    if y > 0:            nbrs.append((x, y-1))
    if y < HEIGHT - 1:   nbrs.append((x, y+1))
    if x > 0:            nbrs.append((x-1, y))
    if x < WIDTH - 1:    nbrs.append((x+1, y))

    uncertain_count = 0
    certain_ns = []

    for nx, ny in nbrs:
        tile = grid[ny][nx]
        if tile == '\0' or not tile:
            uncertain_count += 1
            continue
        # parse n and T safely
        try:
            n_val = int(tile[n_idx])
            T_val = int(tile[T_idx])
        except (IndexError, ValueError):
            # malformed → treat as uncertain for safety
            uncertain_count += 1
            continue

        if T_val < 9:  # certain/trustworthy
            certain_ns.append(n_val)

    if uncertain_count == 0:
        return 0

    avg_n = (sum(certain_ns) / len(certain_ns)) if certain_ns else 0.0
    trust_val = uncertain_count * avg_n

    return '\0' if trust_val > 9 else int(round(trust_val))
        








