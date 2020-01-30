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
from torchvision import datasets, transforms
import win32com.client as wincl

from recycle_models import OtherBestNet, BestSoFarNet


frame_size = 266


def object_detection(vs, fps, firstFrame):
    """
    Use the webcam to determine if any object is in the frame. Once there is
    one, return the frame.
    :return: return code of either 0 (success) or -1 (abort)
    """
    ret = 0
    state = 0
    frame_set = []
    prev_text = ""
    while True:
        text = 'No Object'
        # grab the frame from the threaded video stream and resize it
        # to be the correct frame size for the CNN
        frame = vs.read()
        frame = cv2.resize(frame, (frame_size, frame_size))
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grey = cv2.GaussianBlur(grey, (21, 21), 0)
        
        frameDelta = cv2.absdiff(firstFrame, grey)
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

        if state == 1:  # object is moving, see if it has stopped
            frame_set.append(grey)
            if len(frame_set) >= 100:
                close_count = 0
                for idx, one_frame in enumerate(frame_set):
                    if idx == 0:
                        prev_frame = one_frame
                        continue
                    if np.allclose(prev_frame, one_frame, rtol=2):
                        close_count += 1
                    prev_frame = one_frame
                # if 80 of the last 100 frames are (almost) the same then the
                # object has stopped moving
                if close_count >= 80:
                    text = 'Object Stopped'
                    state = 2
                else:
                    _ = frame_set.pop(0)  # remove the oldest frame
                    
        cv2.imshow("Green Eyes", frame)

        # do not print text to the frame
        # it can interfer with classification due to need to save the image
        if prev_text != text:
            print("[INFO] {}".format(text))
        key = cv2.waitKey(1) & 0xFF
    
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            ret = -1
            break
    
        # if the `g` key was pressed, manual override, send frame
        if key == ord("g"):
            break
        # update the FPS counter
        fps.update()

        # if object has stopped moving save the frame and go to classifier
        if state == 2:
            cv2.imwrite('tempImage/oneImage/frame.jpg', frame)
            break
    return ret


def convert_to_my_classes(x):
    """
    Converts the object classes to recycling classes.
    0: Cardboard -> Grey bin :1
    1: Glass -> Blue bin :0
    2: Metal - > Blue bin :0
    3: Orgranics -> Green bin :3
    4: Paper -> Grey bin :1
    5: Plastic -> Blue bin :0
    6: Trash -> Trash :2
    """
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


def object_classification(model, use_gpu):
    """
    Using the model and the given frame determine what type of recycling is
    shown in the frame.
    :param model: The model to use to classify
    :param use_gpu: Indicates if the GPU is available
    :return: The classification, 0(blue), 1(grey), 2(trash), or 3(green)
    """
    classify_begin = time.time()

    # load the image from the folder to match the process the training used
    transforms_set = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    frame_folder = datasets.ImageFolder('tempImage', transform=transforms_set)
    loader = torch.utils.data.DataLoader(frame_folder, batch_size=1)
    dataiter = iter(loader)
    # get the frame
    frame, _ = dataiter.next()
    # convert the frame to a numpy array
    frame.numpy()
    if use_gpu:
        frame = frame.cuda()
    # get output
    with torch.no_grad():
        output = model(frame)
    print(output)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    print(preds_tensor)
    preds = np.squeeze(preds_tensor.numpy()) if not use_gpu \
        else np.squeeze(preds_tensor.cpu().numpy())
    classify_end = time.time()
    print('[INFO] Classification took {} seconds'.format(classify_end - classify_begin))
    return convert_to_my_classes(preds)


def perform_job(result):
    """
    Given the result from the classification move the platform the correct
    distance and push the recycling into the appropriate bin. The platform
    returns to its original position at the end of this function.
    """
    print('[FUCK] Get moving bitch')
    time.sleep(5)
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

    # load the pretrained model weights
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
    """
    Initialize the video stream
    """
    print("[INFO] Starting video stream...")
    vs = VideoStream(src=1).start()
    time.sleep(2.0)
    fps = FPS().start()
    return vs, fps, 


def main():
    parser = argparse.ArgumentParser(usage='python3 main.py [model to use]')
    parser.add_argument(
        '--model',
        default='OtherBestNet',
        dest='model'
    )
    args = parser.parse_args()
    labels = [
        'blue',
        'grey',
        'trash',
        'green'
    ]

    # set up use of GPU
    use_gpu = torch.cuda.is_available()
    model = model_init(args.model, use_gpu)

    # intialize the webcam
    vs, fps = video_init()

    # set up the speaker
    speaker = wincl.Dispatch("SAPI.SpVoice")

    # first frame for object detection algorithm to use as the background
    firstFrame = vs.read()
    firstFrame = cv2.resize(firstFrame, (frame_size, frame_size))
    firstFrame = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
    firstFrame = cv2.GaussianBlur(firstFrame, (21, 21), 0)
    
    while True:
        # detect if an object is present in the frame and has stopped moving
        ret = object_detection(vs, fps, firstFrame)
        if ret == -1:
            break
        # classify the object in the frame
        result = object_classification(model, use_gpu)
        print('[INFO] Classification:', labels[result])
        speaker.Speak("{} Bin".format(labels[result]))
        # perform the correct action based on the classification
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
