def displayVehicleCount(frame, vehicle_count):
    import cv2
    cv2.putText(
        frame,  # Image
        'Total Detected Vehicles: ' + str(vehicle_count),  # Label
        (20, 20),  # Position
        cv2.FONT_HERSHEY_SIMPLEX,  # Font
        0.8,  # Size
        (0, 0xFF, 0),  # Color
        2,  # Thickness
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
    )


def displayVehicleinFrameCount(frame, vehicle_count):
    import cv2
    cv2.putText(
        frame,  # Image
        'Total Vehicles in frame: ' + str(vehicle_count),  # Label
        (20, 40),  # Position
        cv2.FONT_HERSHEY_SIMPLEX,  # Font
        0.8,  # Size
        (0, 0xFF, 0),  # Color
        2,  # Thickness
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
    )


def boxAndLineOverlap(x_mid_point, y_mid_point, line_coordinates):
    x1_line, y1_line, x2_line, y2_line = line_coordinates

    if (x_mid_point >= x1_line and x_mid_point <= x2_line + 5) and \
            (y_mid_point >= y1_line and y_mid_point <= y2_line + 5):
        return True
    return False


def displayFPS(start_time, num_frames):
    import time
    import os
    current_time = int(time.time())
    if (current_time > start_time):
        os.system('clear')
        print("FPS:", num_frames)
        num_frames = 0
        start_time = current_time
    return start_time, num_frames


def drawDetectionBoxes(indexes, boxes, class_ids, confidences, frame, labels, colors):
    import cv2
    # ensure at least one detection exists
    if len(indexes) > 0:
        # loop over the indices we are keeping
        for i in indexes.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(labels[class_ids[i]],
                                       confidences[i])
            cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # Draw a green dot in the middle of the box
            cv2.circle(frame, (x + (w // 2), y + (h // 2)), 2, (0, 0xFF, 0), thickness=2)


def initializeVideoWriter(video_width, video_height, video_stream, output_video_path):
    import cv2
    # Getting the fps of the source video
    source_video_fps = video_stream.get(cv2.CAP_PROP_FPS)
    # initialize our video writer
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter(output_video_path, fourcc, source_video_fps,
                           (video_width, video_height), True)


def boxInPreviousFrames(previous_frame_detections, current_box, current_detections, frames_before_current):
    import numpy as np
    from scipy import spatial

    centerX, centerY, width, height = current_box
    dist = np.inf  # Initializing the minimum distance
    # Iterating through all the k-dimensional trees
    for i in range(frames_before_current):
        coordinate_list = list(previous_frame_detections[i].keys())
        if len(coordinate_list) == 0:  # When there are no detections in the previous frame
            continue
        # Finding the distance to the closest point and the index
        temp_dist, index = spatial.KDTree(coordinate_list).query([(centerX, centerY)])
        if temp_dist < dist:
            dist = temp_dist
            frame_num = i
            coord = coordinate_list[index[0]]

    if dist > (max(width, height) / 2):
        return False

    # Keeping the vehicle ID constant
    current_detections[(centerX, centerY)] = previous_frame_detections[frame_num][coord]
    return True


def count_vehicles(indexes, boxes, class_ids, vehicle_count, previous_frame_detections, frame, labels, data):
    import cv2
    list_of_vehicles = ["bicycle", "car", "motorbike", "bus", "truck", "train"]
    current_detections = {}
    # ensure at least one detection exists
    if len(indexes) > 0:
        # loop over the indices we are keeping
        for i in indexes.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            centerX = x + (w // 2)
            centerY = y + (h // 2)

            # When the detection is in the list of vehicles, AND
            # it crosses the line AND
            # the ID of the detection is not present in the vehicles
            if labels[class_ids[i]] in data["vehicle_detection"]["list_of_vehicles"]:
                current_detections[(centerX, centerY)] = vehicle_count
                if (not boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections,
                                            data["vehicle_detection"]["frames_before_current"])):
                    vehicle_count += 1
                # vehicle_crossed_line_flag += True
                # else: #ID assigning
                # Add the current detection mid-point of box to the list of detected items
                # Get the ID corresponding to the current detection

                ID = current_detections.get((centerX, centerY))
                # If there are two detections having the same ID due to being too close, 
                # then assign a new ID to current detection.
                if list(current_detections.values()).count(ID) > 1:
                    current_detections[(centerX, centerY)] = vehicle_count
                    vehicle_count += 1

                # Display the ID at the center of the box
                cv2.putText(frame, str(ID), (centerX, centerY), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)

    return vehicle_count, current_detections


def traffic_analyser(filename="Uploaded Videos/bridge.mp4"):
    import cv2
    import numpy as np
    import json
    import time

    with open("config/config.json", 'r+') as f:
        data = json.load(f)

    ## GLOBAL VARIABLES
    # All these classes will be counted as 'vehicles'
    list_of_vehicles = ["bicycle", "car", "motorbike", "bus", "truck", "train"]

    # Setting the threshold for the number of frames to search a vehicle for
    frames_before_current = 10
    input_width, input_height = 416, 416

    # Parse command line arguments and extract the values required
    labels = open(data["args"]["yolo"] + "/coco.names").read().strip().split("\n")
    weights_path = data["args"]["yolo"] + "/yolov3.weights"
    config_path = data["args"]["yolo"] + "/yolov3.cfg"
    input_video_path = filename
    output_video_path = data["args"]["output"]
    pre_defined_confidence = data["args"]["confidence"]
    pre_defined_threshold = data["args"]["threshold"]
    use_gpu = 1

    # Initialize a list of colors to represent each possible class label
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    # load our YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    # Using GPU if flag is passed
    if use_gpu:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # initialize the video stream, pointer to output video file, and
    # frame dimensions
    video_stream = cv2.VideoCapture(input_video_path)
    video_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Specifying coordinates for a default line 
    x1_line = 0
    y1_line = video_height // 2
    x2_line = video_width
    y2_line = video_height // 2

    # Initialization
    previous_frame_detections = [{(0, 0): 0} for i in range(frames_before_current)]
    # previous_frame_detections = [spatial.KDTree([(0,0)])]*frames_before_current # Initializing all trees
    num_frames, vehicle_count = 0, 0
    writer = initializeVideoWriter(video_width, video_height, video_stream, output_video_path)
    start_time = int(time.time())
    end_time = int(time.time())
    max_vehicle_in_frame = 0
    vehicles_in_frame = 0
    # loop over frames from the video file stream
    while True:
        print("================NEW FRAME================")
        num_frames += 1
        print("FRAME:\t", num_frames)
        # Initialization for each iteration
        boxes, confidences, class_ids = [], [], []
        # vehicles_in_frame = 0
        # Calculating fps each second
        # start_time, num_frames = displayFPS(start_time, num_frames)
        # read the next frame from the file
        (grabbed, frame) = video_stream.read()

        # if the frame was not grabbed, then we have reached the end of the stream
        if not grabbed:
            break

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (input_width, input_height),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layer_outputs = net.forward(ln)
        end = time.time()

        # loop over each of the layer outputs
        for output in layer_outputs:
            # loop over each of the detections
            for i, detection in enumerate(output):
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > pre_defined_confidence:
                    # vehicles_in_frame+=1

                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, pre_defined_confidence,
                                   pre_defined_threshold)

        # Draw detection box 
        drawDetectionBoxes(indexes, boxes, class_ids, confidences, frame, labels, colors)

        vehicle_count, current_detections = count_vehicles(indexes, boxes, class_ids, vehicle_count,
                                                           previous_frame_detections, frame, labels, data)
        vehicles_in_frame = len(indexes)
        if vehicles_in_frame > max_vehicle_in_frame:
            max_vehicle_in_frame = vehicles_in_frame
        # Display Vehicle Count if a vehicle has passed the line
        displayVehicleCount(frame, vehicle_count)

        # write the output frame to disk
        writer.write(frame)

        # cv2.imshow('Frame', frame)
        success, jpeg = cv2.imencode('.jpeg', frame)
        im_encoded = jpeg.tobytes()
        im_encoded = b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + im_encoded + b"\r\n\r\n"
        yield im_encoded

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Updating with the current frame detections
        previous_frame_detections.pop(0)  # Removing the first frame from the list
        previous_frame_detections.append(current_detections)
        end_time = int(time.time())
        pd = open("plates.txt", "r")
        summary = {}
        summary["Total Time"] = (end_time - start_time)
        summary["Total Vehicles"] = vehicle_count
        summary["Plates Detected"] = pd.read()
        with open("log.json", "w+") as f:
            json.dump(summary, f)

    writer.release()
    video_stream.release()
