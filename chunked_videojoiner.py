from collections import defaultdict
import os
from glob import glob
from warnings import warn
from argparse import ArgumentParser
import cv2
import json
import datetime

from typing import List, Sequence, Tuple

import numpy as np


        
CURSOR_FILE = os.path.join(os.path.dirname(__file__), "cursors", "mouse_cursor_white_16x16.png")
MINEREC_ORIGINAL_HEIGHT_PX = 720


        
def main(in_path, out_path):
    dataset_path = in_path
    export_path = out_path
    task_id = 0
    cursor_image = cv2.imread(CURSOR_FILE, cv2.IMREAD_UNCHANGED)
    # Assume 16x16
    cursor_image = cursor_image[:16, :16, :]
    cursor_alpha = cursor_image[:, :, 3:] / 255.0
    cursor_image = cursor_image[:, :, :3]
    samples_processed = 0
    # gather all unique IDs for every video/json file pair.
    unique_ids = glob(os.path.join(dataset_path, "*.mp4"))
    unique_ids = list(set([os.path.basename(x).split(".")[0] for x in unique_ids]))
    #self.unique_ids = unique_ids
    frame_size = (640, 360)

    # get all chunked/unchunked trajectories as a dict of lists
    full_trajectory_ids = defaultdict(list)
    for clip_uid in unique_ids:

        # player_uid, game_uid, date, time = clip_uid.split("-")  # doesn't work for "cheeky-cornflower" stuff
        splitted = clip_uid.split("-")
        game_uid, date, time = splitted[-3:]
        player_uid = "-".join(splitted[:-3])

        trajectory_prefix = f"{player_uid}-{game_uid}"
        full_trajectory_ids[trajectory_prefix].append((date, time))
        
        
    # sort and gather the trajectories into a single class
   # trajectories: Sequence[ChunkedContiguousTrajectory] = []


        
            
    for trajectory_prefix in sorted(full_trajectory_ids.keys()):
        date_times = full_trajectory_ids[trajectory_prefix]

        sorted_date_times = list(sorted(date_times))
        video_paths = []
        json_paths = []
        splits = 0
        previous_datetime = None
        i = 0
        while i < len(sorted_date_times):
            date, time = sorted_date_times[i]
            datetime_object = datetime.datetime.strptime(f"{date} {time}", "%Y%m%d %H%M%S")
            if previous_datetime is not None:
                time_difference = datetime_object - previous_datetime
                if time_difference.total_seconds() > (6 * 60):
                    splits += 1
                    new_trajectory_prefix = f"{trajectory_prefix}-{splits}"
                    while i < len(sorted_date_times):
                        date, time = sorted_date_times[i]
                        full_trajectory_ids[new_trajectory_prefix].append((date, time))
                        full_trajectory_ids[trajectory_prefix].remove((date, time))
                        i += 1
                else:
                    path_pref = f"{dataset_path}{trajectory_prefix}-{date}-{time}"
                    outpath = f"{export_path}processed\{trajectory_prefix}-{date}-{time}"

                    video_paths.append(f"{path_pref}.mp4")
                    json_paths.append(f"{path_pref}.jsonl")
                    previous_datetime = datetime_object
                    i += 1
            else:
                path_pref = f"{dataset_path}{trajectory_prefix}-{date}-{time}"
                outpath = f"{export_path}processed\{trajectory_prefix}-{date}-{time}"

                video_paths.append(f"{path_pref}.mp4")
                json_paths.append(f"{path_pref}.jsonl")
                previous_datetime = datetime_object
                i += 1
        
        #assert len(video_paths) == len(json_paths)
        # open the output video/jsonl for write here.
        if os.path.exists(outpath + '.avi'):
            print("Files already exists, skipping...")
            continue
        else:
            vidoutput = cv2.VideoWriter(outpath + '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, frame_size)
            jsonl_out = open(outpath + '.jsonl', 'a')
        print("Completed: " + str(samples_processed) + " samples in total this run. Now Processing: " + path_pref)
        contiguous_clips = []
        for video_path, json_path in zip(video_paths, json_paths):
            #print(json_path)

            if not os.path.exists(video_path) or not os.path.exists(json_path):
                print('error: ' + video_path)
                continue
            
            ##open the video file for writing
            video = cv2.VideoCapture(video_path)
            
            ##open the label file for reading and start processing line by line
            with open(json_path) as json_file:
                #print("pre-processing: " + json_path)
                try:
                    json_lines = json_file.readlines()
                    json_data = "[" + ",".join(json_lines) + "]"
                    json_data = json.loads(json_data)
                except:
                    print("Failed: " + json_path)
                    continue
                    
                ##start processing line by line grabbing video frames and appending cursor to it                 
                for i in range(len(json_data)):
                    step_data = json_data[i]
                    jsonl_out.write(str(json_lines[i])) # write to file
                    
                    # Read frame 
                    ret, frame = video.read()
                    if ret:
                        if step_data["isGuiOpen"]:  #if gui is open, add cursor
                            try:
                                camera_scaling_factor = frame.shape[0] / MINEREC_ORIGINAL_HEIGHT_PX
                                cursor_x = int(step_data["mouse"]["x"] * camera_scaling_factor)
                                cursor_y = int(step_data["mouse"]["y"] * camera_scaling_factor)
                                composite_images_with_alpha(frame, cursor_image, cursor_alpha, cursor_x, cursor_y)
                            except: 
                                print("Failed to add cursor to: " + path_pref)
                                continue
                        #cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
                       # frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
                      #  frame = resize_image(frame, AGENT_RESOLUTION)
                        vidoutput.write(frame)
                    else:
                        print(f"Could not read frame from video {video_path}")
                video.release()
            samples_processed += 1

        vidoutput.release
       # continue
    
    

        #print(full_trajectory_ids[i])

def composite_images_with_alpha(image1, image2, alpha, x, y):
    """
    Draw image2 over image1 at location x,y, using alpha as the opacity for image2.

    Modifies image1 in-place
    """
    ch = max(0, min(image1.shape[0] - y, image2.shape[0]))
    cw = max(0, min(image1.shape[1] - x, image2.shape[1]))
    if ch == 0 or cw == 0:
        return
    alpha = alpha[:ch, :cw]
    image1[y:y + ch, x:x + cw, :] = (image1[y:y + ch, x:x + cw, :] * (1 - alpha) + image2[:ch, :cw, :] * alpha).astype(np.uint8)


        
if __name__ == "__main__":
    parser = ArgumentParser("Run IDM on MineRL recordings.")

    parser.add_argument("--in-path", type=str, required=True, help="Path to minecraft data chunks.")
    parser.add_argument("--out-path", type=str, required=True, help="Path to export to.")

    args = parser.parse_args()

    main(args.in_path,args.out_path)