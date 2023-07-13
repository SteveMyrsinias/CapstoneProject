import torch
import numpy as np
import cv2
from time import time
from ALPR import ocr
import app



def get_video_capture(filename):
    """
    Creates a new video streaming object to extract video frame by frame to make prediction on.
    :return: opencv2 video capture object, with lowest quality frame available for video.
    """

    return cv2.VideoCapture(filename)

def load_model( model_name):
    """
    Loads Yolo5 model from pytorch hub.
    :return: Trained Pytorch model.
    """
    if model_name:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
    else:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def score_frame( frame,model,device):
    """
    Takes a single frame as input, and scores the frame using yolo5 model.
    :param frame: input frame in numpy/list/tuple format.
    :return: Labels and Coordinates of objects detected by model in the frame.
    """
    model.to(device)
    frame = [frame]
    results = model(frame)
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cord

def class_to_label(x,classes):
    """
    For a given label value, return corresponding string label.
    :param x: numeric label
    :return: corresponding string label
    """
    return classes[int(x)]

def createPadding( src, top, bottom, left, right):
    '''
    will alter each frame to add padding to the left so there is constant space for plate ocr
    used to avoid flickering
    will also pad enough so that all sizes of plates can appear
    in case no plate is detected or the confidence is very low it will remain black
     '''
    padded = cv2.copyMakeBorder(src=src,
                                top=top,
                                bottom=bottom,
                                left=left,
                                right=right,
                                borderType=cv2.BORDER_CONSTANT,
                                value=[0, 0, 0]
                                )
    return padded

def plot_boxes( results, frame, c, classes):
    """
    Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
    :param results: contains labels and coordinates predicted by model on the given frame.
    :param frame: Frame which has been scored.
    :return: Frame with bounding boxes and labels ploted on it.
    """
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    if n > 0:
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0] * x_shape - 5), int(row[1] * y_shape - 10), int(
                    row[2] * x_shape + 5), int(
                    row[3] * y_shape + 10)
                bgr = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1)
                cv2.putText(frame, class_to_label(labels[i],classes), (x1, y1), cv2.FONT_HERSHEY_TRIPLEX, 0.9, bgr, 2)
                cropped = frame[y1:y2, x1:x2]

                ocr_results = ocr.main(cropped, c)

                ocr_results = createPadding(ocr_results, 0, frame.shape[0] - ocr_results.shape[0], 0,
                                                 656 - ocr_results.shape[1])
                frame = np.hstack((frame, ocr_results))
                c += 1
            else:
                frame = createPadding(frame, 0, 0, 0, 656)
    else:
        frame = createPadding(frame, 0, 0, 0, 656)

    return frame, c

def test(model_name='C:\\Users\\MyrsiniasS\\OneDrive - Titan Cement Company SA\\Desktop\\pythonProject\\Traffic_monitor\\ALPR\\best.pt',filename="C:\\Users\\MyrsiniasS\\OneDrive - Titan Cement Company SA\\Desktop\\pythonProject\\Traffic_monitor\\ALPR\\IMG_8716.mp4"):
    """
    This function is called when class is executed, it runs the loop to read the video frame by frame,
    and write the output into a new file.
    :return: void
    """

    # capture_index = capture_index
    model = load_model(model_name)
    video = get_video_capture(filename)
    classes = model.names
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using Device: ", device)

    # videolink = app.getVideo()
    cap = video
    #print(videolink)

    c = 0
    i = 0
    gif = []
    #print("Checkpoint")
    while True:
        i += 1
        try:
            (success, frame) = cap.read()
            assert success
            # ret, jpeg = cv2.imencode('.jpeg', frame)
            # imEncoded = jpeg.tobytes()
            # imEncoded = b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + imEncoded + b"\r\n\r\n"
            # yield (imEncoded)
        except:
            print('video ended')
            break

        frame = ocr.resize(frame, 50)

        start_time = time()
        results = score_frame(frame,model,device)

        frame, c = plot_boxes(results, frame, c,classes)

        end_time = time()
        fps = 1 / np.round(end_time - start_time, 2)

        # cv2.putText(frame, f'FPS: {int(fps)}', (20, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)
        # cv2.putText(frame, f'Frame: {int(i)}', (100, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)
        # cv2.imshow('Licence Plate Recognition', frame)

        success, jpeg = cv2.imencode('.jpeg', frame)
        imEncoded = jpeg.tobytes()
        imEncoded = b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + imEncoded + b"\r\n\r\n"
        yield (imEncoded)

        if cv2.waitKey(20) & 0xFF == ord('\x1b'):
            print('video stopped')
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test()
