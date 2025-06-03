import sys
import cv2
from time import time

from trackerCustom import TrackerSiamAttention


def demo(model):
    #Load Model
    if model:
        tracker = TrackerSiamAttention(model) 
  
    # Select ROI
    cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cv2.namedWindow(model, cv2.WND_PROP_FULLSCREEN)
    success, frame = cap.read()

    select_flag = False 
    while success and not select_flag:
        x, y, w, h = cv2.selectROI(model, frame, False, False)
        if w and h :
            select_flag = True
            box = [x, y, w, h]
        print (x, y, w, h)

    tracker.init(frame, box)
    
    while True:
        success, frame = cap.read()
        if not success: break

        pred = tracker.update(frame)

        cv2.rectangle(frame, (int(pred[0]), int(pred[1])), (int(pred[0] + pred[2]), int(pred[1] + pred[3])), (0, 255, 255), 3)
        cv2.imshow(model, frame)

        # Interaction during video
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):break
        elif key == ord('p'): cv2.waitKey()
        elif key == ord('s'): cv2.imwrite("./{}.png".format(time()), frame)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    demo("./pretrained/siamattention/model_e15.pth")