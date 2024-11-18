import os

import cv2
from alive_progress import alive_bar

# image_folder = '/Volumes/Results/dataloc/pv50-nu4-urlx.c0sat600.T170/CM_q_w25_s10_l0to7_1600_1900_strength_t0.712'
# image_folder = '/Volumes/Data/dataloc/pv50-nu4-urlx.c0sat400.T170_highres/imgs'
IFOLDER = "../../../output/ERA5/Y2010-DJFM-daily-NH-850K/img/pv/"
VNAME = IFOLDER + "video.mp4"
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

images = [img for img in os.listdir(IFOLDER) if img.endswith(".png")]
images.sort()
frame = cv2.imread(os.path.join(IFOLDER, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(VNAME, FOURCC, 10, (width, height))

with alive_bar(len(images), force_tty=True) as bar:
    for image in images:
        image_path = os.path.join(IFOLDER, image)
        image = cv2.imread(image_path)
        resized = cv2.resize(image, (width, height))
        video.write(resized)
        bar()

cv2.destroyAllWindows()
video.release()
