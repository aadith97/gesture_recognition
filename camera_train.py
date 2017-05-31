import pygame.camera
import pygame.image
import sys

import crop_face

import cv2



camera_port = 0

ramp_frames = 30

camera = cv2.VideoCapture(camera_port)


i=1

while True :

    retval, im = camera.read()

    cv2.imshow("face",im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #photo=pygame.image.save(img, 'dataset_image/1/image'+str(i)+'.jpg')

    try:
            if(i<=50):

                resized_image = cv2.resize(im, (64, 64))
                cv2.imwrite('dataset_image/5/image'+str(i)+'.jpg', resized_image)
                i=i+1
            else:
                break
    except:
        print('1')


