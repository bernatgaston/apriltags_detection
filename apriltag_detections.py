import cv2
import numpy as np
from apriltag import apriltag


def get_frame(mirror=False):
    ret_val, color = cam.read()
    if mirror: 
        img = cv2.flip(color, 1)
    # converting to gray-scale
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    return color, gray

def detect_tag(image):
    #imagepath = 'tag36h11.png'
    #image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    #print(image)
    detector = apriltag("tag36h11")
    detections = detector.detect(image)
    # for i in detections:
    #     pose, e0, e1 = detector.detection_pose(detections, (3156.71852, 3129.52243, 359.097908, 239.736909), 0.0762)
    
    print(detections)
    return detections

def show_image(frame, detections):
    # obtain corners
    if detections:
        lb = [int(detections[0]['lb-rb-rt-lt'][0][0]), int(detections[0]['lb-rb-rt-lt'][0][1])]
        rb = [int(detections[0]['lb-rb-rt-lt'][1][0]), int(detections[0]['lb-rb-rt-lt'][1][1])]
        rt = [int(detections[0]['lb-rb-rt-lt'][2][0]), int(detections[0]['lb-rb-rt-lt'][2][1])]
        lt = [int(detections[0]['lb-rb-rt-lt'][3][0]), int(detections[0]['lb-rb-rt-lt'][3][1])]
        # draw the bounding box of the AprilTag detection
        cv2.line(frame, lb, rb, (0, 255, 0), 4)
        cv2.line(frame, rb, rt, (0, 255, 0), 4)
        cv2.line(frame, rt, lt, (0, 255, 0), 4)
        cv2.line(frame, lt, lb, (0, 255, 0), 4)
    # Show image
    cv2.imshow('my webcam', frame)

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    while True:
        color, gray = get_frame()
                #Detect tags
        detections = detect_tag(gray)
        show_image(color, detections)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()