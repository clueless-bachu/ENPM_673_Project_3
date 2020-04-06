import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_frame(filepath, frame_no, lent = False):
    '''
    Return an image in the nth frame of a video sequence

    input:
    filepath: the relative or absolute path of the video file
    frame_no: The nth frame we want to extract

    return:
    a cv2 image matrix of the nth frame of the video sequence
    '''
    cap = cv2.VideoCapture(filepath) #video is the video being called
    if lent:
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return length
    else:
        cap.set(1,frame_no); # Where frame_no is the frame you want
        ret, frame = cap.read() # Read the frame
        return frame


def prepare_data(video_path, path_to_data,train_images= 100):
    length = extract_frame(video_path, 1, True)
    for i in range(0, length, length//train_images):
        frame = extract_frame(video_path, i)
        cv2.imwrite(path_to_data+"{}.png".format(i), frame)

video_path = r'.\data\detectbuoy.avi'
img_path = r'.\data\img'
prepare_data(video_path, img_path)
