from utils import get_center_of_bbox, get_bbox_width, get_foot_position
from ultralytics import YOLO
import supervision as sv  # run tracker after predict
import os
import pickle
import sys
import cv2
import pandas as pd
import numpy as np
sys.path.append('./')
# import pickle


class Tracker:
    def __init__(self, model_path):  # do when call
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        for object, obkect_tracks in tracks.items():
            for frame_num, track in enumerate(obkect_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    if object == "ball":
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]["position"]= position
        
        return tracks
            
    def interpolate_ball_position(self, ball_positions):
        ball_positions = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(
            ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}}
                          for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            # we predict cuz we need to change the ID of the goalkeeper
            detections_batch = self.model.predict(
                frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referee": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # convert GK to player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if (cls_names[class_id] == "goalkeeper"):
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]
            # track objects
            detection_with_track = self.tracker.update_with_detections(
                detection_supervision)

            tracks["players"].append({})
            tracks["referee"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_track:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                tracker_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][tracker_id] = {"bbox": bbox}

                if cls_id == cls_names_inv["referee"]:
                    tracks["referee"][frame_num][tracker_id] = {"bbox": bbox}

            # the ball is unique we dont need to track
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center-rectangle_width//2
        x2_rect = x_center+rectangle_width//2
        y1_rect = int((y2-rectangle_height//2)+21)
        y2_rect = int((y2+rectangle_height//2)+21)
        if track_id is not None:
            cv2.rectangle(frame, (x1_rect, y1_rect),
                          (int(x2_rect), y2_rect), color, cv2.FILLED)
            cv2.putText(frame, f"{track_id}", (x1_rect + 12, y1_rect+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_point = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20]
        ])
        cv2.drawContours(frame, [triangle_point], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_point], 0, color, 2)

        return frame

    def draw_annotations(self, videos_frames, tracks, team_ball_control):
        cnt_team1 = 0
        cnt_team2 = 0
        output_video_frames = []
        for frame_num, frame in enumerate(videos_frames):
            frame = frame.copy()
            
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referee"][frame_num]

            # Draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(
                    frame, player["bbox"], color, track_id)

                if player.get("has_ball", False):
                    frame = self.draw_triangle(
                        frame, player["bbox"], (0, 0, 255))
            # Draw referee
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(
                    frame, referee["bbox"], (0, 225, 255), track_id)
             # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))
                
            # ball possesion
            if team_ball_control[frame_num] == 1:
                cnt_team1 = cnt_team1 + 1
            else:
                if team_ball_control[frame_num] == 2:
                    cnt_team2 = cnt_team2 + 1
                    
            team1 = (cnt_team1 )/(cnt_team1+cnt_team2)
            team2 = (cnt_team2 )/(cnt_team1+cnt_team2)
            cv2.putText(frame, f"Team 1 ctrl: {team1*100:.2f}%", ( 1400,900),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            cv2.putText(frame, f"Team 2 ctrl: {team2*100:.2f}%", ( 1400,950),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3) 
            #draw team 
            output_video_frames.append(frame)

        return output_video_frames
