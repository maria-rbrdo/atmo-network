import cv2
import os
from alive_progress import alive_bar

image_folder = '../../dataloc/pv50-nu4-urlx.c0sat600.T85/imgs'
video_name = 'video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))

with alive_bar(len(images), force_tty=True) as bar:
    for image in images:
        image_path = os.path.join(image_folder, image)
        image = cv2.imread(image_path)
        resized = cv2.resize(image, (width, height))
        video.write(resized)
        bar()

cv2.destroyAllWindows()
video.release()
