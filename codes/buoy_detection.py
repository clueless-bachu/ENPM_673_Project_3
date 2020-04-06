import numpy as np
import cv2
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import json

means = np.load('means.npy')
covariances = np.load('covar.npy')

filepath = r"../project3/data/detectbuoy.avi"
# out = cv2.VideoWriter('buoy_detection.mp4', cv2.VideoWriter_fourcc(*'MP4V') , 15, (640,480))
cap = cv2.VideoCapture(filepath)
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
while(cap.isOpened()):
    ret, img_ = cap.read()
    if ret:
        img = img_[150:380]
        blur = (cv2.GaussianBlur(img,(7,7),1.42))
        edged = cv2.Canny(blur,155,250)

        detected_circles = cv2.HoughCircles(edged,  
                            cv2.HOUGH_GRADIENT, 1.5, 120, param1 = 1000, 
                       param2 = 19, minRadius = 10, maxRadius = 75)  
        if detected_circles is not None: 

            detected_circles = np.uint16(np.around(detected_circles)) 

            for pt in detected_circles[0, :]: 
                a, b, r = pt[0], pt[1], pt[2]
                color = np.argmax([
                    multivariate_normal(means[0], covariances[0]).pdf(img[b,a]),
                    multivariate_normal(means[1], covariances[1]).pdf(img[b,a]),
                    multivariate_normal(means[2], covariances[2]).pdf(img[b,a]),
                ])
                if color == 0:
                    cv2.circle(img, (a, b), r, (0, 255, 0), 2)
                elif color == 1:
                    cv2.circle(img, (a, b), r, (0, 69, 255), 2)
                elif color ==2:
                    cv2.circle(img, (a, b), r, (0, 255, 255), 2)
        img_[150:380] = img
        cv2.imshow('detection', img_)
#         out.write(img_)
#         sleep(0.1)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
# out.release()
cap.release()
cv2.destroyAllWindows()