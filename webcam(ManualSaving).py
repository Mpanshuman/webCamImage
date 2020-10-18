import cv2 as cv
import time
import os

webcam = cv.VideoCapture(0)
webcam.set(3, 640) # set width
webcam.set(4, 480) # set height
webcam.set(10, 100) # set brightness
counter = 1

## path for saving file

name_dir = 'Anshuman'
parent_dir = r"C:\Users\Anshuman\Desktop\openCvProject"
Img_directory = name_dir
save_img_path = os.path.join(parent_dir,Img_directory)

try:
    os.mkdir(save_img_path)
except OSError as error:
    print(error)

faceCascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

while webcam.isOpened():
    flag, image = webcam.read()
    k = cv.waitKey(1)

    # Face Detection

    faces = faceCascade.detectMultiScale(image,1.1,4)

    # if face is not detected

    # if faces is ():
    #     print('No face found')

    for (x,y,w,h) in faces:
        cv.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
    
    cv.imshow('webcam', image)
    
    # press 'Q' to quit camera

    if k == ord('q') or counter == 10:
        break
    
    # press 'S' to Save Image (max 10 pic)
    elif k == ord('s'):
        
        # change directory to the directory where the image will be saved 
        os.chdir(save_img_path)
        
        cv.imwrite(f'image{counter}.png',image)
        print(f'Saving image{counter}')
        counter +=1
                
        if counter == 10:
            print('Data Collected')
            break
            


webcam.release()
cv.destroyAllWindows()