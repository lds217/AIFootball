from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance import SpeedAndDistance_Estimator

def main():
    # read video
    video_frames = read_video('inputvideo/08fd33_4.mp4')

    # Initialize tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    tracker.add_position_to_tracks(tracks)
    # cam move track
    cam_move = CameraMovementEstimator(video_frames[0])
    cam_move_per_frame = cam_move.get_camera_movement(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/cam_move_stubs.pkl')
    
    cam_move.adjust_position_to_tracks(tracks,cam_move_per_frame)
    
    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    # Interpolate ball pos

    tracks["ball"] = tracker.interpolate_ball_position(tracks["ball"])

    # save cropped image of a player
    # print(tracks["players"][1])
    for track_id, player in tracks["players"][1].items():
        bbox = player["bbox"]
        frame = video_frames[0]

        # crop from frame
        cropped_image = frame[int(bbox[1]):int(
            bbox[3]), int(bbox[0]):int(bbox[2])]

        # save the cropped image
        cv2.imwrite(f'outputvideo/cropped_img.jpg', cropped_image)
        break

    # save_video(video_frames, 'outputvideo/output1.avi')
    #  player_dict = tracks["players"][frame_num]for track_id, player in player_dict.items():
    
      # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    
    # team assigner
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])

    # tracks["players"][frame_num][tracker_id] = {"bbox": bbox}
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track["bbox"], player_id)
            tracks["players"][frame_num][player_id]['team'] = team
            tracks["players"][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # assigned player ball
        player_assigner =PlayerBallAssigner()
    team_ball_control= []
    # team_ball_control.append(0)
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(0)

    print(team_ball_control)

    # draw output
    output_video_frames = tracker.draw_annotations(video_frames, tracks , team_ball_control)

    #draw cam move
    output_video_frames = cam_move.draw_camera_movement(output_video_frames,cam_move_per_frame)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    # save video
    save_video(output_video_frames, 'outputvideo/output3.avi')


if __name__ == '__main__':
    main()
