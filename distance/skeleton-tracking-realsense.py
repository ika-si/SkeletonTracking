#!/usr/bin/env python3
from collections import namedtuple
import util as cm
import cv2
import pyrealsense2 as rs
import math
import numpy as np
from skeletontracker import skeletontracker

from Person_reID_pytorch import re_identification_realsense
import calibration
import measure_distance
import save_frame
import open_sound_protcol


def render_ids_3d(
    render_image, skeletons_2d, depth_map, depth_intrinsic, joint_confidence
):


#    thickness = 1
#    text_color = (255, 255, 255)
    rows, cols, channel = render_image.shape[:3]
    distance_kernel_size = 10
    
    pos_list = [[0 for j in range(3)] for i in range(len(skeletons_2d))]
    reid_list = [0 for j in range(len(skeletons_2d))]
    
    pos_x = 0
    pos_y = 0
    pos_z = 0
            
    # calculate 3D keypoints and display them
    for skeleton_index in range(len(skeletons_2d)):
        skeleton_2D = skeletons_2d[skeleton_index]
        joints_2D = skeleton_2D.joints
        did_once = False
        
        reid_list[skeleton_index] = re_identification_realsense.re_identification(skeleton_index)
            
        for joint_index in range(len(joints_2D)):
                
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
                    '''
                    cv2.putText(
                        render_image,
                        str(point_3d),
                        (int(joints_2D[joint_index].x), int(joints_2D[joint_index].y)),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.4,
                        color=(255,0,0),
                        thickness=2,
                    )
                    '''
                    
                       
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

                        pos_list[skeleton_index][0] = pos_x
                        pos_list[skeleton_index][1] = pos_y
                        pos_list[skeleton_index][2] = pos_z

            if did_once == False:
                cv2.putText(
                    color_image,
                    "id:" + str(skeleton_index),
                    (int(joints_2D[0].x), int(joints_2D[0].y - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    color=(0,225,0),
                    thickness=2,
                )
                did_once = True
    '''  
    print(pos_list)
    print()
    '''
    calibration_pos_list = calibration.calibration_pos(pos_list)
    open_sound_protcol.show_color_osc(calibration_pos_list, reid_list)
    distance_list = measure_distance.measure_diff(calibration_pos_list)
    open_sound_protcol.change_particles(distance_list, reid_list)

# Main content begins
if __name__ == "__main__":
    try:
        
        idx = 0
        n = 0
        
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
        
        #REIDを呼び出す
        re_identification_realsense.model_load()
        
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
            
            # save frame
            idx += 1
            if idx == 30:
                frame_list = save_frame.save_frame_camera_key(color_image, 'data/temp', "capture", skeletons)
                #print(frame_list)
                re_identification_realsense.pred_person(frame_list)
                idx = 0    

            
            cv2.imshow(window_name, color_image)
            
            #全体画像保存
#            n = save_frame_camera_key(color_image, 'data/temp', 'camera_capture', n)

            
            if cv2.waitKey(1) == 27:
                break
 
        pipeline.stop()
        cv2.destroyAllWindows()

    except Exception as ex:
        print('Exception occured: "{}"'.format(ex))
