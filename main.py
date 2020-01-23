import sys
import argparse
import time
import torch
import numpy as np
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import time
import cv2

from recycle_models import OtherBestNet, BestSoFarNet


frame_size = 266


def object_detection(vs, fps, firstFrame):
    """
    Use the webcam to determine if any object is in the frame. Once there is
    one, return the frame.
    :return: frame containing an object to classify as a numpy array
    """
    ret = 0
    state = 0
    frame_set = []
    while True:
        text = 'No Object'
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = cv2.resize(frame, (frame_size, frame_size))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
               cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
    
        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 150:
                state = 0
                frame_set = []
                continue
            text = 'Object Moving'
            state = 1

        if state == 1:
            frame_set.append(gray)
            if len(frame_set) >= 100:
                close_count = 0
                for idx, one_frame in enumerate(frame_set):
                    if idx == 0:
                        prev_frame = one_frame
                        continue
                    if np.allclose(prev_frame, one_frame, rtol=1):
                        close_count += 1
                    prev_frame = one_frame
                if close_count >= 80:
                    text = 'Object Stopped'
                    state = 2
                else:
                    _ = frame_set.pop(0)  # remove the oldest frame
                    
        cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (255, 255, 255), 2)
        cv2.imshow("Green Eyes", frame)
        key = cv2.waitKey(1) & 0xFF
    
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            ret = -1
            break
    
        # update the FPS counter
        fps.update()

        # if object has stopped moving send frame to the classifier
        if state == 2:
            break

    return frame, ret


def convert_to_my_classes(x):
    my_dict = [
        [1, 2, 5],  # blue
        [0, 4],  # grey
        [6],  # trash
        [3],  # green
    ]
    if x in my_dict[0]:
        return 0
    elif x in my_dict[1]:
        return 1
    elif x in my_dict[2]:
        return 2
    elif x in my_dict[3]:
        return 3
    else:
        print('[ERROR] No valid class')
        return -1


def object_classification(model, frame, use_gpu):
    """
    Using the model and the given frame determine what type of recycling is
    shown in the frame.
    :param model: The model to use to classify
    :param frame: The frame containing an object to classify
    :param use_gpu: Indicates if the GPU is available
    :return: The classification, 0(blue), 1(grey), 2(trash), or 3(green)
    """
    classify_begin = time.time()
    frame = frame.transpose(2, 0, 1)
    frame = torch.tensor(frame, dtype=torch.float)
    frame = frame.reshape(1, 3, frame_size, frame_size)
    if use_gpu:
        frame = frame.cuda()
    # get sample outputs
    output = model(frame)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy()) if not use_gpu \
        else np.squeeze(preds_tensor.cpu().numpy())
    classify_end = time.time()
    print(preds)
    print('[INFO] Classification took {} seconds'.format(classify_end - classify_begin))
    return convert_to_my_classes(preds)


def perform_job(result):
    """
    Given the result from the classification move the platform the correct
    distance and push the recycling into the appropriate bin. The platform
    returns to its original position at the end of this function.
    """
    print('[FUCK] Get moving bitch')
    #time.sleep(5)
    print('[FUCK] Okay done')
    pass


def model_init(model_name, use_gpu):
    """
    Initialize the model to use, using the saved pretrained models.
    """
    if model_name == 'BestSoFarNet':
        model = BestSoFarNet()
    elif model_name == 'OtherBestNet':
        model = OtherBestNet()
    else:
        print('[ERROR] Invalid model given {}'.format(model_name))
        sys.exit(0)

    if use_gpu:
        print('[INFO] Using GPU')
        model.cuda()
        model.load_state_dict(torch.load('modelPaths/{}.'
                                         'pth'.format(model_name.lower())))
    else:
        print('[INFO] Using CPU')
        model.load_state_dict(torch.load('modelPaths/{}.'
                                         'pth'.format(model_name.lower()),
                                          map_location=torch.device('cpu')))
    model.eval()
    return model

def video_init():
    print("[INFO] Starting video stream...")
    vs = VideoStream(src=1).start()
    time.sleep(2.0)
    fps = FPS().start()
    return vs, fps, 


def main():
    parser = argparse.ArgumentParser(usage='python3 main.py [model to use]')
    parser.add_argument(
        '--model',
        default='BestSoFarNet',
        dest='model'
    )
    args = parser.parse_args()
    # Set up use of GPU
    use_gpu = torch.cuda.is_available()
    model = model_init(args.model, use_gpu)
    vs, fps = video_init()
    firstFrame = vs.read()
    firstFrame = cv2.resize(firstFrame, (frame_size, frame_size))
    firstFrame = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
    firstFrame = cv2.GaussianBlur(firstFrame, (21, 21), 0)
    
    while True:
        frame, ret = object_detection(vs, fps, firstFrame)
        if ret == -1:
            break
        result = object_classification(model, frame, use_gpu)
        print('[INFO] Classification:', result)
        perform_job(result)

    print('[INFO] Terminating...')
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] Elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))
    
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    return 0


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('[ERROR] Aborting')
    sys.exit(0)
