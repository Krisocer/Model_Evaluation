import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import numpy as np
from PIL import Image
import json
import csv
import collections
from step_utils import image_paths, calculate_iou
import cv2
import pandas as pd

def read_ground_truth_labels(frame_name, annotations_path):
    with open(annotations_path, "r") as f:
        labels = json.load(f)
    
    for label in labels:
        if label["name"] == frame_name:
            return label["labels"]

    return []

def accuracy(y_true, y_pred):
    if not y_true and not y_pred:
        return 1
    if not y_true:
        return 0

    y_true_counter = collections.Counter(y_true)
    y_pred_counter = collections.Counter(y_pred)
    return sum((y_true_counter & y_pred_counter).values()) / sum(y_true_counter.values())

def count_accuracy(y_true, y_pred):
    return 1 - abs(y_true - y_pred) / y_true

def get_all_image_paths(root_path):
    paths = []
    for subdir, _, files in os.walk(root_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                paths.append(os.path.join(subdir, file))
    return paths

def calculate_and_write_averages(video_id):
    csv_file = f"results/{video_id}.csv"
    data = pd.read_csv(csv_file)
    
    avg_object_count = data['Object Counts'].mean()
    avg_accuracy = data['Accuracy'].mean()
    avg_binary_accuracy = data['Binary Accuracy'].mean()
    avg_iou = data['IoU Results'].apply(eval).apply(lambda x: np.mean(x) if x else np.nan).mean()

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Averages", avg_object_count, avg_accuracy, "", avg_binary_accuracy, avg_iou])


def write_to_csv(video_id, frame_name, object_counts, acc, binary_decision, binary_acc, iou_results):
    csv_file = f"results/{video_id}.csv"
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    headers = ['Frame', 'Object Counts', 'Accuracy', 'Binary Decision', 'Binary Accuracy', 'IoU Results']
    row = [frame_name, object_counts, acc, binary_decision, binary_acc, iou_results]

    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

if __name__ == "__main__":
    image_root_path = "train/images/"
    annotations_folder = "train/annotations/"

    # Capture path of those keyframes
    paths = get_all_image_paths(image_root_path)

    # Load yolov5
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    model = model.cuda()
    
    video_ids = set([os.path.dirname(p).split("/")[-1] for p in paths])

    for video_id in video_ids:
        total_object_count = 0
        total_accuracy = 0
        total_binary_accuracy = 0
        total_iou_sum = 0
        total_frames = 0

        video_paths = [p for p in paths if os.path.dirname(p).split("/")[-1] == video_id]

        for current_frame_path in video_paths:
            frame_name = os.path.basename(current_frame_path)
            gt_labels = read_ground_truth_labels(frame_name, os.path.join(annotations_folder, f"{video_id}.json"))

            current_frame = Image.open(current_frame_path)
            temp_res = model(current_frame)
            temp_info = temp_res.pandas().xyxy[0]

            y_true_labels = [l['category'] for l in gt_labels]
            y_pred_labels = temp_info['name'].tolist()

            # Calculate IoU for ground truth and predicted bounding boxes
            y_pred_boxes = []
            for _, row in temp_info.iterrows():
                y_pred_boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            gt_boxes = [[l['box2d']['x1'], l['box2d']['y1'], l['box2d']['x2'], l['box2d']['y2']] for l in gt_labels]

            iou_results = []
            for gt_box in gt_boxes:
                max_iou = 0
                for pred_box in y_pred_boxes:
                    iou = calculate_iou(gt_box, pred_box)
                    max_iou = max(max_iou, iou)
                iou_results.append(max_iou)

            # Write a count for the number of objects of a given type that appear and the accuracy
            object_counts = collections.Counter(y_true_labels)
            object_counts = sum(object_counts.values())  # Updated to only include the number of objects
            acc = accuracy(y_true_labels, y_pred_labels)

            # A binary decision regarding whether an object of a given type appears in a frame and the accuracy
            unique_true_labels = np.unique(y_true_labels)
            binary_y_true = [1 if l in y_true_labels else 0 for l in unique_true_labels]
            binary_y_pred = [1 if l in y_pred_labels else 0 for l in unique_true_labels]
            binary_decision = dict(zip(unique_true_labels, binary_y_true))
            binary_acc = accuracy(binary_y_true, binary_y_pred)
            
            # Increment accumulators
            total_object_count += object_counts
            total_accuracy += acc
            total_binary_accuracy += binary_acc
            total_iou_sum += sum(iou_results)
            total_frames += 1

            # Write the information to a CSV file
            write_to_csv(video_id, frame_name, object_counts, acc, binary_decision, binary_acc, iou_results)

        # Write the average values to the last row of the CSV file
        avg_object_count = total_object_count / total_frames
        avg_accuracy = total_accuracy / total_frames
        avg_binary_accuracy = total_binary_accuracy / total_frames
        avg_iou = total_iou_sum / total_frames

        # Write the averages for the current video
        calculate_and_write_averages(video_id)

"""
if __name__ == "__main__":
    # Set up here
    image_path = "results/keyframes/"
    confidence_threshould = 0.5

    # Capture path of those keyframes
    paths = image_paths(image_path)

    # Load yolov5
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    model = model.cuda()

    # Save information of each keyframe
    total_info_keyframes = []
    for current_frame_path in paths:
        current_frame = Image.open(current_frame_path)
        temp_res = model(current_frame)
        temp_info = temp_res.pandas().xyxy[0]
        total_info_keyframes.append(temp_info)
        # temp_box_x1 = temp_info['xmin']
        # temp_box_y1 = temp_info['ymin']
        # temp_box_x2 = temp_info['xmax']
        # temp_box_y2 = temp_info['ymax']
        # temp_class = temp_info['class']
        # temp_name = temp_info['name']
        # temp_confidence = temp_info['confidence']

    # Start Evaluation
    final_stats = []
    previous_frame_info = None
    for current_frame_info in total_info_keyframes:
        if previous_frame_info is None:
            previous_frame_info = current_frame_info
            continue

        temp_stas_info = {}

        # Regularization 1
        # The variance of the number of detected objects
        previous_num_objects = sum(previous_frame_info['confidence'] >= confidence_threshould)
        current_num_objects = sum(current_frame_info['confidence'] >= confidence_threshould)
        temp_stas_info['num_objects_change_compared_to_previous'] = current_num_objects - previous_num_objects

        # Regularization 2
        # The move of each object should be big enough, which means the IoU should be expected to be small
        # We calculate the IoU of the each object here
        previous_all_boxes = []
        for i in range(len(previous_frame_info['xmin'])):
            temp_box = []
            temp_box.append(previous_frame_info['xmin'][i])
            temp_box.append(previous_frame_info['ymin'][i])
            temp_box.append(previous_frame_info['xmax'][i])
            temp_box.append(previous_frame_info['ymax'][i])
            previous_all_boxes.append(temp_box)

        current_all_box = []
        for i in range(len(current_frame_info['xmin'])):
            temp_box = []
            temp_box.append(current_frame_info['xmin'][i])
            temp_box.append(current_frame_info['ymin'][i])
            temp_box.append(current_frame_info['xmax'][i])
            temp_box.append(current_frame_info['ymax'][i])
            current_all_box.append(temp_box)

        box1 = previous_all_boxes
        box2 = current_all_box
        if previous_num_objects > current_num_objects:    # we need len(box1) <= len(box2) for calculation
            box1 = current_all_box
            box2 = previous_all_boxes
        iou_info = []
        for e in box1:
            temp_iou = 0.0
            for f in box2:
                temp_iou_here = calculate_iou(e, f)
                temp_iou = temp_iou_here if temp_iou_here > temp_iou else temp_iou
            iou_info.append(temp_iou)

        temp_stas_info['all_iou_info'] = iou_info

        # ==================
        # Add to final stats
        final_stats.append(temp_stas_info)

    # Open csv file in write mode and write the headers
    with open('stats.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame Number', 'Num Objects', 'Variance of # Objects', 'Num Zero IoU', 'Average Non-zero IoU'])

        # Write the info for each frame into the csv file
        for i, e in enumerate(final_stats):
            t1 = e['num_objects_change_compared_to_previous']
            t2 = e['all_iou_info']

            zero_iou_count = 0
            non_zero_iou_count = []
            for f in t2:
                if f == 0:
                    zero_iou_count += 1
                else:
                    non_zero_iou_count.append(f)

            writer.writerow([i+2, len(t2), t1, zero_iou_count, '{:.2f}'.format(np.average(non_zero_iou_count))])   

    
    # Print the info stats table
    for i, e in enumerate(final_stats):
        t1 = e['num_objects_change_compared_to_previous']
        t2 = e['all_iou_info']

        print("=====================")

        print("# of frame: {}".format(i+2))

        print("# of objects: {}".format(len(t2)))

        print("Variance of #. objects: {}".format(t1))

        # print("IoU info of each object: {}".format(t2))

        zero_iou_count = 0
        non_zero_iou_count = []
        for f in t2:
            if f == 0:
                zero_iou_count += 1
            else:
                non_zero_iou_count.append(f)
        print("#. of zero IoU: {}".format(zero_iou_count))

        print('Average of non-zero IoU: {:.2f}'.format(np.average(non_zero_iou_count)))
"""

