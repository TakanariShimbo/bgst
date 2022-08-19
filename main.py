import os
import cv2
import numpy as np


# setting
filename = 'test_1'

# define opening
# kernel = np.ones((5,5),np.uint8)
def customized_opening(img, scale, iteration):
    img = cv2.medianBlur(img, 5)
    for i in range(iteration):
        img = cv2.resize(img, dsize=None, fx=1/scale, fy=1/scale)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img


# define bgst model
# Ref: https://pystyle.info/opencv-background-substraction/
bgst = cv2.createBackgroundSubtractorMOG2(
    history=15, varThreshold=50, detectShadows=False
)


# read movie
video_path = "../datas/original_movies/{}.mp4".format(filename)
cap = cv2.VideoCapture(video_path)

# make directory for save
image_folder_path = "../datas/original_images/{}/".format(filename)
os.makedirs(image_folder_path, exist_ok=True)

mask_folder_path = "../datas/mask_images/{}/".format(filename)
os.makedirs(mask_folder_path, exist_ok=True)

# main loop
cnt = 0
while True:
    # get frame
    ret, frame = cap.read()
    if not ret:
        break
    if cnt > 400:
        break

    # original movie is 120fps -> convert 30fps
    if cnt%4 == 0:
        # get mask
        mask = bgst.apply(frame)

        # opening
        # opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
        mask = customized_opening(img=mask, scale=2, iteration=5)
        mask[mask != 0] = 255

        # mix frame and mask
        frame[mask == 0] = 0

        # viz
        cv2.imshow("Frame (Only Forground)", frame)
        cv2.waitKey(1)

        # save
        cv2.imwrite(image_folder_path + "frame{}.jpg".format(cnt), frame)
        cv2.imwrite(mask_folder_path + "mask{}.jpg".format(cnt), mask)
    cnt += 1

cap.release()
cv2.destroyAllWindows()