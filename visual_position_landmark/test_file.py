#!/usr/bin/python3

import cv2
import time
from build_lib.run_parameter import*
# args = get_input()
# # print("arguments", args)
# cap = cv2.VideoCapture(0)
# while True:
#     ret, img = cap.read()
#     time.sleep(0.1)
#     save_pred_(cap, img, args)


#     print("Running")
#     #cv2.imshow("test", img)
#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


cap = cv2.VideoCapture("1_104_60.mp4")
while True:

            # Đợi và lấy frame từ camera
            #frames = pipeline.wait_for_frames()
            _, frames = cap.read()
            cv2.imshow("video", frames)
            if cv2.waitKey(1) == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()
