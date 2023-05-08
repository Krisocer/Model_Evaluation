import torch
import numpy as np
import cv2
import os

from step_utils import img_ssim

# **************** Introduction ***********************
# This script file is used for generating the keyframes
# according to the given ssim (as threshold)
# *****************************************************


if __name__ == "__main__":
    # Set up here
    video_path = "test_videos/000f8d37-d4c09a0f.mov"
    ssim_threshold = 0.1
    print("current ssim threshold is " + str(ssim_threshold))

    # Load video
    cap = cv2.VideoCapture(video_path)
    
    if fps == 0:
        print("This video is not available")

    # Set output path
    keyframePath = "results/keyframes"
    if not os.path.exists(keyframePath):
        os.makedirs(keyframePath)

    # Read the first frame.
    ret, prev_frame = cap.read()

    i = 0
    count = 0
    while ret:
        ret, curr_frame = cap.read()
        if ret:
            # Compute ssim between two adjacent frames
            ssim = img_ssim(prev_frame, curr_frame)

            # If meets requirement
            if ssim > ssim_threshold:
                print("Saving Frame # {}".format(i), end='\r')
                cv2.imwrite('results/keyframes/' + str(i) + '.jpg', curr_frame)
                count += 1

            prev_frame = curr_frame
            i += 1
    print("Total Number of frames saved: {}".format(count))