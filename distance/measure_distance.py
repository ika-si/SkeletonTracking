import numpy as np
from scipy.special import comb

SOCIAL_DISTANCE = 200

def measure_diff(coord_set):
    try:
        coord_length = len(coord_set)
        distance_list = [SOCIAL_DISTANCE for j in range(coord_length)]
    
        for i in range(0, coord_length-1):
        
            diff = []
        
            coord0 = np.array(coord_set[i])
        
            for j in range(i+1, coord_length):
                coord1 = np.array(coord_set[j])
            
                coord2 = coord0-coord1
                distance = np.sqrt(coord2[0]**2 + coord2[2]**2)
                distance = np.floor(distance)
                diff.append(distance)
            
                distance_list[j] = min(distance_list[j], diff[j-i-1])
#               print(j-i-1)
#               print(distance_list)
            
            min_diff = min(diff)
            distance_list[i] = min(distance_list[i], min_diff)
        
        for i in range(0, coord_length):
            if coord_set[i][0] == 0 and coord_set[i][2] == 0:
                distance_list[i] = 0
                
#        print(distance_list)
        
        return distance_list
    
    except Exception as ex:
        print('Exception occured: "{}"'.format(ex))

'''
if __name__ == "__main__":
    
    coord_set = [
        # uvz(screen cood), xyz(unrealengine coord)
        [370, 130, 320],
        [0, 130, 0],
        [0, 130, 0],
        [0, 130, 0]
    ]

    distance_list = measure_distance(coord_set)
    print(distance_list)
'''