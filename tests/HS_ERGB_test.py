import pathlib
import cv2
import os
from typing import List

### IMPORTED FROM PROJECT ###
from tools import zipped

### CONSTANTS ####
TARGET_FPS = 60 # FPS after interpolation
HS_ERGB_RGB_FILE_TEMPLATE = "{:06d}.png"
RGB_FOLDER_FILTER = "images_corrected"
HS_ERGB_EVENT_FILE_TEMPLATE = "{:06d}.npz"
EVENT_FOLDER_FILTER = "events_aligned"

def process_HS_ERGB_dataset(hs_ergb_path: pathlib.Path) -> None:
    # Given path to HS_ERGB dataset, interpolate each frame live based on event data
    print("Obtaining image and event folder names...")
    image_folders = zipped.read_dataset_directory(archive=hs_ergb_path, filter=RGB_FOLDER_FILTER)
    event_folders = zipped.read_dataset_directory(archive=hs_ergb_path, filter=EVENT_FOLDER_FILTER)
    print("Obtained image and event folder names.")
    assert(len(image_folders) == len(event_folders))

    print(f"Image folders: {image_folders}")
    for curr_image_folder in image_folders:
        image_pattern = "hsergb/" + curr_image_folder + "/%06d.png"
        event_pattern = ""

        print(f"Image pattern: {image_pattern}")
        cap = cv2.VideoCapture(image_pattern)
        while True:
            # Read current frame from native video
            ret, ref_frame = cap.read()

            # Stream ended
            if not ret:
                print(f"Video {curr_image_folder} has ended. Breaking stream.")
                break

            # Extract all events occuring between ref_frame and next frame
            #ref_events = pass

            # Generate a new frame based on the reference frame and 
            #interpolate()

            cv2.imshow("Frame", ref_frame)
            cv2.waitKey(1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    hs_ergb_folder = pathlib.Path("hsergb")
    process_HS_ERGB_dataset(hs_ergb_folder)