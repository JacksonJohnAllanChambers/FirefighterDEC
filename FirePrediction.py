## Fire Prediction Code 
## Store data from any values >0  (Fire #, Position x, Position y, Data)

# for string xxxyyfwcf
# x cord (0-2)
# y cord (3-4)
# fire (6)
# severity (7)
# wind (8)
# direction (9)
# citizens (10)
# firefighters (11)


current_map = [['\0' for _ in range(550)] for _ in range(100)]
predicted_map = [['\0' for _ in range(550)] for _ in range(100)]

def main():
    for x in range(550):
        for y in range (100):
            ## See if there is a fire, if the severity is greater than 4, and if the windspeed is greater than 0
            if get_fire(x, y) == 1 & get_severity(x,y) >= 4 & get_wind(x,y) > 0:
                        ## Find the wind direction & set new fire adjacent to wind direction
                        match get_direction(x,y):
                            case 0:
                                #Check if there is already a fire at position
                                if(get_fire(x, y - 1) == 1):
                                    new_severity(x, y - 1, get_severity(x, y - 1) + 1)
                                #if not a new fire set a new one
                                else:
                                    new_fire(x, y - 1)
                                    new_severity(x, y - 1, get_severity(x,y)/2)
                            case 1:
                                if(get_fire(x + 1, y) == 1):
                                    new_severity(x + 1, y, get_severity(x + 1, y) + 1)
                                else:
                                    new_fire(x + 1, y)
                                    new_severity(x + 1, y, get_severity(x,y)/2)
                            case 2:
                                if(get_fire(x, y + 1) == 1):
                                    new_severity(x, y + 1, get_severity(x, y - 1) + 1)
                                else:
                                    new_fire(x, y + 1)
                                    new_severity(x, y + 1, get_severity(x,y)/2)
                            case 3:
                                if(get_fire(x - 1, y) == 1):
                                    new_severity(x - 1, y, get_severity(x - 1, y) + 1)
                                else:
                                    new_fire(x - 1, y)
                                    new_severity(x - 1, y, get_severity(x,y)/2)
    return

#Getter
def get_fire(x,y):
    return current_map[x][y][6]
def get_wind(x,y):
    return current_map[x][y][8]
def get_direction(x,y):
    return current_map[x][y][9]
def get_severity(x,y):
    return current_map[x][y][7]

#Setter
def new_fire(x,y,n):
    predicted_map[x][y][6] = n
    return
def new_severity(x,y,n):
    predicted_map[x][y][7] = n
    return

set(predicted_map)