import cv2 as cv
import numpy as np
import os

BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

image_width = 368
image_height = 368
threshold = 0.2

# load the OpenPose model
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

# set up camera capture
cap = cv.VideoCapture(4)

# create a folder to save annotated images
output_folder = "annotated_images"
os.makedirs(output_folder, exist_ok=True)

# create a file to store labels
label_file_path = "labels.txt"

frame_count = 0

while True:
    # capture frame-by-frame
    ret, img = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    photo_height, photo_width = img.shape[:2]

    # prepare the image for OpenPose
    blob = cv.dnn.blobFromImage(img, 1.0, (image_width, image_height), (127.5, 127.5, 127.5), swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()
    out = out[:, :19, :, :]

    # process the output
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (photo_width * point[0]) / out.shape[3]
        y = (photo_height * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > threshold else None)

    # flag to determine if a pose is drawn
    pose_drawn = False

    # draw the pose pairs
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv.line(img, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(img, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(img, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            pose_drawn = True

    # display the resulting frame
    cv.imshow('Pose Detection', img)

    # save the annotated frame
    frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
    cv.imwrite(frame_filename, img)

    # write label to the file
    with open(label_file_path, 'a') as label_file:
        label_file.write(f"{frame_filename}, {'Pose Drawn' if pose_drawn else 'No Pose'}\n")

    frame_count += 1

    # break the loop on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# elease the camera and close all windows
cap.release()
cv.destroyAllWindows()
