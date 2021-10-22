#!/usr/bin/env python3
from collections import namedtuple
import util as cm
import cv2
import os
import time
import pyrealsense2 as rs
import math
import numpy as np
from skeletontracker import skeletontracker
from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder

capture_number = 0
capture_flag = True

def render_ids_3d(
    render_image, skeletons_2d, depth_map, depth_intrinsic, joint_confidence
):

    thickness = 1
    text_color = (255, 255, 255)
    rows, cols, channel = render_image.shape[:3]
    distance_kernel_size = 10
    
    global capture_flag
    global capture_number
    
    if capture_number != len(skeletons_2d):
        capture_flag = True
    
    capture_number = len(skeletons_2d)
           
    # calculate 3D keypoints and display them
    for skeleton_index in range(len(skeletons_2d)):
        skeleton_2D = skeletons_2d[skeleton_index]
        joints_2D = skeleton_2D.joints
        #print(skeleton_2D.id)
        #print(len(joints_2D))
        did_once = False
        
        # save frame
        save_frame_camera_key(render_image, 'data/temp', 'camera_capture', skeleton_index, joints_2D)
        
        for joint_index in range(len(joints_2D)):
            if did_once == False:
#                cv2.putText(
#                    render_image,
#                    "id: " + str(skeleton_2D.id),
#                    (int(joints_2D[joint_index].x), int(joints_2D[joint_index].y - 30)),
#                    cv2.FONT_HERSHEY_SIMPLEX,
#                    0.55,
#                    text_color,
#                    thickness,
#                )
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
#                    cv2.putText(
#                        render_image,
#                        str(point_3d),
#                        (int(joints_2D[joint_index].x), int(joints_2D[joint_index].y)),
#                        cv2.FONT_HERSHEY_DUPLEX,
#                        0.4,
#                        text_color,
#                        thickness,
#                    )

        

        #print(len(skeletons_2d))
        
    capture_flag = False
        

def save_frame_camera_key(color_image, dir_path, basename, person_id, joints_2D, ext='jpg', delay=1):
    
    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)
    print(base_path)
    
    
#   key = cv2.waitKey(delay) & 0xFF
#   if key == ord('c'):
    
    global capture_flag
    print(capture_flag)
    if capture_flag:
        
        y1 = int(joints_2D[0].y)
        y2 = int(joints_2D[10].y)
        x1 = int(joints_2D[4].x)
        x2 = int(joints_2D[7].x)
    
#       print(joints_2D)
    
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
#            cm.render_result(skeletons, color_image, joint_confidence)
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
