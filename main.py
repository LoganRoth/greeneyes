import sys
import argparse
import torch
import numpy as np

from recycle_models import OtherBestNet, BestSoFarNet


def object_detection():
    """
    Use the webcam to determine if any object is in the frame. Once there is
    one, return the frame.
    :return: frame containing an object to classify as a numpy array
    """
    # TODO: Add code for detecting an object, should return the frame
    #       containing the object as a numpy array
    return np.zeros((10, 10))


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
        print('Error, no valid class')
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
    if use_gpu:
        frame = torch.tensor(frame).cuda()
    # get sample outputs
    output = model(frame)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy()) if not use_gpu \
        else np.squeeze(preds_tensor.cpu().numpy())
    return preds[0]


def perform_job(result):
    """
    Given the result from the classification move the platform the correct
    distance and push the recycling into the appropriate bin. The platform
    returns to its original position at the end of this function.
    """
    print('Get moving bitch')
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
        print('Invalid model given {}'.format(model_name))
        sys.exit(0)

    if use_gpu:
        print('Using GPU')
        model.cuda()
        model.load_state_dict(torch.load('modelPaths/{}.'
                                         'pth'.format(model_name.lower())))
    else:
        print('Using CPU')
        model.load_state_dict(torch.load('modelPaths/{}.'
                                         'pth'.format(model_name.lower()),
                                         map_location=torch.device('cpu')))
    model.eval()
    return model


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
    while True:
        frame = object_detection()
        result = object_classification(model, frame, use_gpu)
        print('Classification:', result)
        perform_job(result)
    return 0


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Aborting')
    sys.exit(0)
