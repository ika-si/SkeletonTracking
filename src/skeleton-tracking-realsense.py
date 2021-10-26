#!/usr/bin/env python3
from collections import namedtuple
import util as cm
import cv2
import os
import pyrealsense2 as rs
import math
import numpy as np
from scipy.special import comb
from skeletontracker import skeletontracker
from pythonosc import udp_client

from pythonosc.osc_message_builder import OscMessageBuilder
from Person_reID_pytorch import re_identification_realsense

#最大人数
Human_Number = 5

IP = '192.168.0.24'
#大学
#IP = '172.20.61.72'

capture_number = 0
capture_flag = True

#real_distance_realsense_width = 450
real_distance_realsense_width = 30
SOCIAL_DISTANCE = 500

def save_frame_camera_key(color_image, dir_path, basename, person_id, joints_2D, ext='jpg', delay=1):

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)


#   key = cv2.waitKey(delay) & 0xFF
#   if key == ord('c'):

    global capture_flag
    print(capture_flag)
    if capture_flag:
        
        y1 = int(joints_2D[0].y)
        y2 = int(joints_2D[10].y)
        x1 = int(joints_2D[4].x)
        x2 = int(joints_2D[7].x)
    
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
                print("----------------------------------")
                h, w = save_image.shape[:2]
                height = round(h * (50 / w))
                resize_image = cv2.resize(save_image, dsize=(50, height))
                cv2.imwrite('{}_{}.{}'.format(base_path, person_id, ext), resize_image)
            except Exception as ex:
                print("imwrite error")

def show_color_osc(distance_list):
    
    try:
        for i in range(0, Human_Number):
            PORT = 10000 + i

            # UDPのクライアントを作る
            client = udp_client.UDPClient(IP, PORT)

            # メッセージを作って送信する
            msg = OscMessageBuilder(address='/pos')
            msg.add_arg(distance_list[i][0])
            msg.add_arg(distance_list[i][2])
            m = msg.build()

            client.send(m)
    except Exception:
        pass

def measure_distance_osc(distance_list):    
    
    
    try:
        # case = comb(n, r)
        case = comb(Human_Number, 2, exact=True)
    
        diff_distance = [SOCIAL_DISTANCE for j in range(case)]
        
        '''
        for i in range(0, Human_Number-1):
            # x　メートル換算
            x1 = real_distance_realsense_width/1280*(distance_list[i][0]-640)
            #x1 = distance_list[i][0]-640
            if distance_list[i][2]**2-x1**2 > 0:
                d1 = (distance_list[i][2]**2-x1**2)**0.5
            else:
                d1 = distance_list[i][2]
            for j in range(i+1, Human_Number):
                x2 = real_distance_realsense_width/1280*(distance_list[j][0]-640)
                #x2 = distance_list[j][0]-640
                if distance_list[j][2]**2-x2**2 > 0:
                    d2 = (distance_list[j][2]**2-x2**2)**0.5
                else:
                    d2 = distance_list[j][2]
                
                distance = ((x1 - x2)**2 + (d1 - d2)**2)**0.5
                
                if(type(distance) is complex):
                    print("complex")
                else:
                    #diff_distance[(i+j+2)%3] = distance
            
                    if i==0:
                        diff_distance[0] = min(diff_distance[0] , distance)
                  
                    if i == 1 or j == 1:
                        diff_distance[1] = min(diff_distance[1], distance)
                    
                    if j == 2:
                        diff_distance[2] = min(diff_distance[2], distance)
                    
        '''
        
        rad1 = 0
        rad2 = 0
        
        if distance_list[2][2] == 0:
            Detected_Human_Number = 2
        elif distance_list[1][2] == 0:
            Detected_Human_Number = 1
        else:
            Detected_Human_Number = 3
        
        #余弦定理
        for i in range(0, Detected_Human_Number-1):
            # x　メートル換算
            x1 = real_distance_realsense_width/1280*abs(distance_list[i][0]-640)
            #x1 = distance_list[i][0]-640
            d1 = distance_list[i][2]
            
            if d1 > 0:
                rad1 = math.acos(x1/d1)

            for j in range(i+1, Detected_Human_Number):
                x2 = real_distance_realsense_width/1280*abs(distance_list[j][0]-640)
                #x2 = distance_list[j][0]-640
                d2 = distance_list[j][2]
                
                if d2 > 0:
                    rad2 = math.acos(x2/d2)
                
                if 180-rad1-rad2 > 0 and rad1 != 0 and rad2 != 0:
                    distance = (d1**2 + d2**2 - 2*d1*d2*math.cos(180-rad1-rad2))**0.5
                
                    print(distance)
                    
                    if(type(distance) is complex):
                        print("complex")
                    else:
                    #diff_distance[(i+j+2)%3] = distance
            
                        if i==0:
                            diff_distance[0] = min(diff_distance[0] , distance)
                  
                        if i == 1 or j == 1:
                            diff_distance[1] = min(diff_distance[1], distance)
                    
                        if j == 2:
                            diff_distance[2] = min(diff_distance[2], distance)
                            
                            
        if distance_list[2][2] == 0:
            diff_distance[2] = 0
        if distance_list[1][2] == 0:
            diff_distance[1] = 0
        if distance_list[0][2] == 0:
            diff_distance[0] = 0

        print(diff_distance)
        
        #TouchDesignerへ  
        for i in range(0, 3):
            
            PORT = 1100 + i

            # UDPのクライアントを作る
            client = udp_client.UDPClient(IP, PORT)

            # メッセージを作って送信する
            msg = OscMessageBuilder(address='/dis')
            
            msg.add_arg(diff_distance[i])
                
            m = msg.build()
             
            client.send(m)
                

    
    except (TypeError, NameError):
        print(TypeError)
        print(NameError)
        
        
def render_ids_3d(
    render_image, skeletons_2d, depth_map, depth_intrinsic, joint_confidence
):


    thickness = 1
    text_color = (255, 255, 255)
    rows, cols, channel = render_image.shape[:3]
    distance_kernel_size = 10
    
    distance_list = [[0 for j in range(3)] for i in range(Human_Number)]
    pos_x = 0;
    pos_y = 0;
    pos_z = 0;
    
           
    # calculate 3D keypoints and display them
    for skeleton_index in range(len(skeletons_2d)):
        skeleton_2D = skeletons_2d[skeleton_index]
        joints_2D = skeleton_2D.joints
        did_once = False
            
        re_id = re_identification_realsense.re_identification(skeleton_index);
        #print(skeleton_index, "     ", re_id)
            
        for joint_index in range(len(joints_2D)):
            if did_once == False:
                cv2.putText(
                    render_image,
                    "id: " + str(skeleton_2D.id),
                    (int(joints_2D[joint_index].x), int(joints_2D[joint_index].y - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    text_color,
                    thickness,
                )
                did_once = True
                
            # check if the joint was detected and has valid coordinate
            if skeleton_2D.confidences[joint_index] > joint_confidence:
                distance_in_kernel = []
                low_bound_x = max(
                    0,
                    int(
                        joints_2D[joint_index].x - math.floor(distance_kernel_size / 2)
                    ),
                )
                upper_bound_x = min(
                    cols - 1,
                    int(joints_2D[joint_index].x + math.ceil(distance_kernel_size / 2)),
                )
                low_bound_y = max(
                    0,
                    int(
                        joints_2D[joint_index].y - math.floor(distance_kernel_size / 2)
                    ),
                )
                upper_bound_y = min(
                    rows - 1,
                    int(joints_2D[joint_index].y + math.ceil(distance_kernel_size / 2)),
                )
                for x in range(low_bound_x, upper_bound_x):
                    for y in range(low_bound_y, upper_bound_y):
                        distance_in_kernel.append(depth_map.get_distance(x, y))
                
                # depth
                median_distance = np.percentile(np.array(distance_in_kernel), 50)
                # x, y
                depth_pixel = [
                    int(joints_2D[joint_index].x),
                    int(joints_2D[joint_index].y),
                ]
                
                if median_distance >= 0.3:
                    point_3d = rs.rs2_deproject_pixel_to_point(
                        depth_intrinsic, depth_pixel, median_distance
                    )
                    point_3d = np.round([float(i) for i in point_3d], 3)
                    point_str = [str(x) for x in point_3d]
                    cv2.putText(
                        render_image,
                        str(point_3d),
                        (int(joints_2D[joint_index].x), int(joints_2D[joint_index].y)),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.4,
                        text_color,
                        thickness,
                    )
                    
                       
                    if skeleton_2D.confidences[1] > joint_confidence:

                        # x, y, z
                        pos_x = round(joints_2D[1].x, 2)

                        pos_y = round(joints_2D[1].y, 2)
                        
                        pos_z = round(
                                depth_map.get_distance(
                                    int(joints_2D[1].x), int(joints_2D[1].y)
                                    )*100, 2
                                )
                        
#                        print(pos_x, "       ", pos_z)
                        
                        
                        distance_list[re_id][0] = pos_x;
                        distance_list[re_id][1] = pos_y;
                        distance_list[re_id][2] = pos_z;
#                        distance_list[skeleton_index][3] = re_id;
                        
    show_color_osc(distance_list)
    measure_distance_osc(distance_list)



# Main content begins
if __name__ == "__main__":
    try:
        # Configure depth and color streams of the intel realsense
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        # Start the realsense pipeline
        pipeline = rs.pipeline()
        pipeline.start()

        # Create align object to align depth frames to color frames
        align = rs.align(rs.stream.color)

        # Get the intrinsics information for calculation of 3D point
        unaligned_frames = pipeline.wait_for_frames()
        frames = align.process(unaligned_frames)
        depth = frames.get_depth_frame()
        depth_intrinsic = depth.profile.as_video_stream_profile().intrinsics

        # Initialize the cubemos api with a valid license key in default_license_dir()
        skeletrack = skeletontracker(cloud_tracking_api_key="")
        joint_confidence = 0.2

        # Create window for initialisation
        window_name = "cubemos skeleton tracking with realsense D400 series"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO)
        
        # Change window size
        #cv2.resizeWindow(window_name, 1600, 900)
        
        #REIDを呼び出す
        #ReID.pred_person()

        while True:
            # Create a pipeline object. This object configures the streaming camera and owns it's handle
            unaligned_frames = pipeline.wait_for_frames()
            frames = align.process(unaligned_frames)
            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
            if not depth or not color:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth.get_data())
            color_image = np.asanyarray(color.get_data())

            # perform inference and update the tracking id
            skeletons = skeletrack.track_skeletons(color_image)

            # render the skeletons on top of the acquired image and display it
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            cm.render_result(skeletons, color_image, joint_confidence)
            render_ids_3d(
                color_image, skeletons, depth, depth_intrinsic, joint_confidence
            )
            cv2.imshow(window_name, color_image)
            
            if cv2.waitKey(1) == 27:
                break
 
        pipeline.stop()
        cv2.destroyAllWindows()

    except Exception as ex:
        print('Exception occured: "{}"'.format(ex))
