# **************** Introduction ***********************
# This script file is used for generating the keyframes
# according to the given ssim (as threshold)
# *****************************************************
import re
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
    

def process_video_folder(video_folder, output_folder, ssim_threshold):
    image_paths = sorted([os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith(('.jpg', '.png'))])
    if not image_paths:
        print(f"No images found in the folder: {video_folder}")
        return

    prev_frame = cv2.imread(image_paths[0])
    gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    os.makedirs(output_folder, exist_ok=True)
    cv2.imwrite(os.path.join(output_folder, os.path.basename(image_paths[0])), prev_frame)
    saved_frames_count = 1

    for path in image_paths[1:]:
        frame = cv2.imread(path)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sim = ssim(gray_prev_frame, gray_frame, full=True)[0]

        if sim < ssim_threshold:
            cv2.imwrite(os.path.join(output_folder, os.path.basename(path)), frame)
            saved_frames_count += 1
            prev_frame = frame
            gray_prev_frame = gray_frame
    print(f"Total saved frames for video_id {os.path.basename(video_folder)}: {saved_frames_count}")
    
    
if __name__ == "__main__":
    image_folder_path = "/home/krisocer/steps/train/images/"
    output_base_folder = "results/keyframes"
    start_threshold = 0.4
    end_threshold = 0.6
    interval = 0.05

    ssim_thresholds = np.arange(start_threshold, end_threshold + interval, interval)

    video_folders = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) if os.path.isdir(os.path.join(image_folder_path, f))]

    for ssim_threshold in ssim_thresholds:
        print(f"Processing SSIM threshold: {ssim_threshold:.2f}")
        threshold_folder = os.path.join(output_base_folder, str(ssim_threshold))
        for video_folder in video_folders:
            video_id = os.path.basename(video_folder)
            if video_id.startswith('.') or not re.match(r'\w{8}-\w{8}', video_id):
                continue
            print(f"Processing video_id: {video_id}")
            output_folder = os.path.join(threshold_folder, video_id)
            process_video_folder(video_folder, output_folder, ssim_threshold)

'''
if __name__ == "__main__":
    # Set up here
    video_path = "test_videos/000f8d37-d4c09a0f.mov"
    ssim_threshold = 0.5
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
    
'''