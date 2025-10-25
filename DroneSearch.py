import Mapping
import fireClasses

quit = 0

x = fireClasses.drone.x
y = fireClasses.drone.y
x_limit = 0
y_limit = 0

highFire = 0

Drone_map = [['\0' for _ in range(x_limit)] for _ in range(y_limit)]

def Search_Fire():
    if quit == 1:
        return
    while x_limit <= x <= x_limit + 1:
        while y_limit <= y <= y_limit + 1:
            fire = Mapping.get_fire(x,y)
            if fire >= highFire:
                highFire = (x,y)
            if highFire != 0:
                Mapping.move(highFire)
            else:
                Search_Fire

def Empty_Search():
    Mapping.get_info(x,y)
