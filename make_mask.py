import os
import cv2
import glob
import tqdm

# setting
filename = 'test_1'

# define opening
# kernel = np.ones((5,5),np.uint8)
def customized_opening(img, scale, iteration):
    for i in range(iteration):
        img = cv2.medianBlur(img, 5)
        img = cv2.resize(img, dsize=None, fx=1/scale, fy=1/scale)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img


# define bgst model
# Ref: https://pystyle.info/opencv-background-substraction/
bgst = cv2.createBackgroundSubtractorMOG2(
    history=15, varThreshold=50, detectShadows=False
)

# make directory for save
mask_folder_path = "../datas/mask_images/{}/".format(filename)
os.makedirs(mask_folder_path, exist_ok=True)

# read images
image_folder_path = "../datas/original_images/{}/".format(filename)
total_frame = len(glob.glob(image_folder_path+'*.bmp'))

for i in tqdm.tqdm(range(total_frame)):
    image_path = image_folder_path + 'frame{}.bmp'.format(i)
    img_bgr = cv2.imread(image_path)

    img_mask_gray = bgst.apply(img_bgr)

    # opening
    # opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
    img_mask_gray = customized_opening(img=img_mask_gray, scale=2, iteration=10)
    img_mask_gray[img_mask_gray != 0] = 255

    # viz
    img_viz_bgr = img_bgr.copy()
    img_viz_bgr[img_mask_gray == 0] = 0
    cv2.imshow("Frame (Only Forground)", img_mask_gray)
    cv2.waitKey(1)

    # save
    cv2.imwrite(mask_folder_path + "frame{}.bmp".format(i), img_mask_gray)