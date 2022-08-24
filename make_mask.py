import argparse
import glob
import os

import cv2
import tqdm


# define opening
# kernel = np.ones((5,5),np.uint8)
def customized_opening(img, scale, iteration):
    for i in range(iteration):
        img = cv2.medianBlur(img, 3)
        img = cv2.resize(img, dsize=None, fx=1/scale, fy=1/scale)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img


def make_mask(opt):
    # define bgst model
    # Ref: https://pystyle.info/opencv-background-substraction/
    bgst = cv2.createBackgroundSubtractorMOG2(
        history=15, varThreshold=50, detectShadows=False
    )

    # save directory
    save_folder_path = f"../datas/mask_images/{opt.filename}/"
    os.makedirs(save_folder_path, exist_ok=True)

    # read directory
    image_folder_path = f"../datas/original_images/{opt.filename}/"
    total_frame = len(glob.glob(image_folder_path+'*.bmp'))

    for i_frame in tqdm.tqdm(range(total_frame)):
        # read images
        image_path = image_folder_path + f"frame{i_frame}.bmp"
        img_bgr = cv2.imread(image_path)

        img_mask_gray = bgst.apply(img_bgr)

        # opening
        # opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
        img_mask_gray = customized_opening(img=img_mask_gray, scale=2, iteration=10)
        img_mask_gray[img_mask_gray != 0] = 255

        # save
        save_path = save_folder_path + f"frame{i_frame}.bmp"
        cv2.imwrite(save_path, img_mask_gray)

        # viz
        if opt.show_viz:
            img_viz_bgr = img_bgr.copy()
            img_viz_bgr[img_mask_gray == 0] = 0
            cv2.imshow("Masked Frame", img_viz_bgr)
            cv2.waitKey(1)


if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)
    parser.add_argument('--show_viz', type=bool)
    opt = parser.parse_args()
    print(opt)

    # run main
    make_mask(opt)

    # ex)
    # python make_mask.py --filename "test_1"