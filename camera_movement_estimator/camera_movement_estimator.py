import pickle
import cv2
import numpy as np
import sys
import os
sys.path.append('./')
from utils import mdistance

class CameraMovementEstimator():
    def __init__(self, frame):
        
        self.mindis = 5
        
        first_frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:,0:20] =1
        mask_features[:,900:1050] = 1
        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7,
            mask = mask_features
        )

    def get_camera_movement(self, frames, read_from_stub = False, stub_path = None):
        #Read the stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)
        
        camera_movement = [[0,0]]*len(frames)
        
        
        
        old_gray = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray,**self.features) ## ** to extract the dict into the para
        
        for frame_num in range(1,len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num],cv2.COLOR_BGR2GRAY)
            new_features, _,_ = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_features,None,**self.lk_params) ## lucas kanade params 
            
            max_distance = 0
            camera_movement_x, camera_movement_y = 0,0
            for i, (new,old) in enumerate(zip(new_features,old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel() # flattened_arr 
                
                distance = mdistance(new_features_point,old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x = old_features_point[0] - new_features_point[0]
                    camera_movement_y = old_features_point[1] - new_features_point[1]
            
            if max_distance > self.mindis:
                camera_movement[frame_num] = [camera_movement_x,camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray,**self.features)
                
            old_gray= frame_gray.copy()
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)
        
        return camera_movement
    
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []
        
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            x_move, y_move = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f"X: {x_move:.2f}", ( 10,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv2.putText(frame, f"Y: {y_move:.2f}", ( 10,90),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3) 
            output_frames.append(frame)
        
        return output_frames
    
    def adjust_position_to_tracks(self, tracks, camera_movement_per_frame):
        for object, obkect_tracks in tracks.items():
            for frame_num, track in enumerate(obkect_tracks):
                for track_id, track_info in track.items():
                    position = track_info["position"]
                    cam_move =camera_movement_per_frame[frame_num]
                    position_adj = (position[0]-cam_move[0],position[1]-cam_move[1])
                    tracks[object][frame_num][track_id]["position_adj"]= position_adj
        
        return tracks
            
            