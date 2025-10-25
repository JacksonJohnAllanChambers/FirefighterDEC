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

import Mapping


def fire_prediction(Current_map):
    map = Current_map
    for x in range(550):
        for y in range (100):
            ## See if there is a fire, if the severity is greater than 4, and if the windspeed is greater than 0
            if Mapping.get_fire(x, y) == 1 and Mapping.get_fire(x,y) >= 4 and Mapping.get_wind(x,y) >= 3:
                        ## Find the wind direction & set new fire adjacent to wind direction
                        match Mapping.get_direction(x,y):
                            case 0:
                                #Check if there is already a fire at position
                                if(Mapping.get_fire(x, y - 1) == 1):
                                    Mapping.set_fire(x, y - 1, Mapping.get_fire(x, y - 1) + 1)
                                if(get_fire(x, y - 1) >= 1):
                                    Mapping.set_fire(x, y - 1, get_fire(x, y - 1) + 1)
                                #if not a new fire set a new one
                                else:
                                    new_fire(x, y - 1, get_fire(x,y)/2)
                            case 1:
                                if(Mapping.get_fire(x + 1, y) == 1):
                                    new_severity(x + 1, y, get_severity(x + 1, y) + 1)
                                if(Mapping.get_fire(x + 1, y) >= 1):
                                    Mapping.set_fire(x + 1, y, Mapping.get_fire(x + 1, y) + 1)
                                else:
                                    Mapping.set_fire(x + 1, y, get_fire(x,y)/2)
                            case 2:
                                if(get_fire(x, y + 1) >= 1):
                                    Mapping.set_fire(x, y + 1, get_fire(x, y - 1) + 1)
                                else:
                                    Mapping.set_fire(x, y + 1, get_fire(x,y)/2)
                            case 3:
                                if(Mapping.get_fire(x - 1, y) == 1):
                                    new_severity(x - 1, y, get_severity(x - 1, y) + 1)
                                if(get_fire(x - 1, y) >= 1):
                                    Mapping.set_fire(x - 1, y, get_fire(x - 1, y) + 1)
                                else:
                                    Mapping.set_fire(x - 1, y, get_fire(x,y)/2)
                                    
    Mapping.update_predicted_map(map)
    return