import os
import cv2
import tqdm


# setting
filename = 'test_1'     # string
total_frame = 1001      # int
down_fps_rate = 4       # int

# read movie
video_path = "../datas/original_movies/{}.mp4".format(filename)
cap = cv2.VideoCapture(video_path)

# make directory for save
image_folder_path = "../datas/original_images/{}/".format(filename)
os.makedirs(image_folder_path, exist_ok=True)

# main loop
cnt = 0
with tqdm.tqdm(total=total_frame) as pbar:
    while True:
        # get frame
        ret, img_bgr = cap.read()
        if not ret:
            break
        if cnt >= total_frame*down_fps_rate:
            break

        # save
        if cnt%down_fps_rate == 0:
            cv2.imwrite(image_folder_path + "frame{}.bmp".format(cnt//down_fps_rate), img_bgr)
            pbar.update(1)
        cnt += 1

    cap.release()
    cv2.destroyAllWindows()