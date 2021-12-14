import os
import cv2
from Person_reID_pytorch import re_identification_realsense

#画像格納
frame_list = [0 for j in range(30)]

def save_frame_camera_key(color_image, dir_path, basename, skeletons_2d, ext='png'):
    
    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)
    
    global frame_list
    
    for skeleton_index in range(len(skeletons_2d)):
        skeleton_2D = skeletons_2d[skeleton_index]
        joints_2D = skeleton_2D.joints
        
        y1 = int(joints_2D[1].y)
        y2 = int(joints_2D[10].y)
        x1 = int(joints_2D[2].x)
        x2 = int(joints_2D[5].x)
    
        if(x1 > x2):
            temp = x1
            x1 = x2
            x2 = temp
        if(y1 > y2):
           temp = y1
           y1 = y2
           y2 = y1
           
        gap = 30
        if(y1-gap >= 0):
            y1 = y1 - gap
        if(y2+gap <= 720):
            y2 = y2 + gap
        if(x1-gap >= 0):
            x1 = x1 - gap
        if(x2+gap <= 1280):
            x2 = x2 + gap
        
        if color_image is None:
                return
#        if color_image.all():
        else:
            save_image = color_image[y1:y2, x1:x2]
            try:
                person_id = skeleton_2D.id
                
                h, w = save_image.shape[:2]
                height = round(h * (50 / w))
                resize_image = cv2.resize(save_image, dsize=(50, height))
                resize_image = re_identification_realsense.gamma_processing(resize_image)
                frame_list[skeleton_index] = resize_image
                print("---------------- save picture ------------------")
                #print(frame_list)
                
            except Exception as ex:
#                print('Exception occured: "{}"'.format(ex))
                print("---------------- error ------------------")

    return frame_list
                
                
                