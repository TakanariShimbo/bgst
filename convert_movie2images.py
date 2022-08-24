import argparse
import os

import cv2
import tqdm


def convert_movie2images(opt):
    # read movie
    video_path = f"../datas/original_movies/{opt.filename}.mp4"
    cap = cv2.VideoCapture(video_path)

    # save directory
    image_folder_path = f"../datas/original_images/{opt.filename}/"
    os.makedirs(image_folder_path, exist_ok=True)

    with tqdm.tqdm(total=opt.total_frame) as pbar:
        cnt = 0
        while True:
            # get frame
            ret, img_bgr = cap.read()
            if not ret:
                break
            if cnt >= opt.total_frame * opt.down_fps_rate:
                break

            # save
            if cnt%opt.down_fps_rate == 0:
                # resize
                img_bgr = cv2.resize(img_bgr, dsize=None, fx=1/opt.down_resolution_rate, fy=1/opt.down_resolution_rate)

                # save
                save_path = image_folder_path + f"frame{cnt//opt.down_fps_rate}.bmp"
                cv2.imwrite(save_path, img_bgr)

                # viz
                if opt.show_viz:
                    cv2.imshow("Original Frame", img_bgr)
                    cv2.waitKey(1)

                pbar.update(1)
            cnt += 1

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)
    parser.add_argument('--total_frame', type=int)
    parser.add_argument('--down_fps_rate', type=int, default=1.0)
    parser.add_argument('--down_resolution_rate', type=float, default=1.0)
    parser.add_argument('--show_viz', type=bool)
    opt = parser.parse_args()
    print(opt)

    # run main
    convert_movie2images(opt)

    # ex)
    # python convert_movie2images.py --filename "test_1" --total_frame 1001 --down_fps_rate 4 --down_resolution_rate 3