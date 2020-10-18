import cv2 as cv
import numpy as np
from PIL import Image
import os

# Returns Cropped Image/Frame

def image_collect(frame):
    faces = faceCascade.detectMultiScale(frame,1.1,4)

    if faces is ():
        return None
    
    for (x,y,w,h) in faces:
        
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face = frame[y:y+h,x:x+w]

    return cropped_face

# Saves Image in train and test folder

def save_image(counter , image):

    if counter <= 50:
        os.chdir(train_img)
    else:
        os.chdir(test_img)
      
    
    cv.imwrite(f'{counter}.png',image)
    print(f'Saving Image{counter}')


webcam = cv.VideoCapture(0)
webcam.set(3, 640) # set width
webcam.set(4, 480) # set height
webcam.set(10, 100) # set brightness
counter = 0

## path for saving file

parent_dir = r"C:\Users\Anshuman\Desktop\sambhu_opencv"
Img_directory = 'Anshuman'
save_img_path = os.path.join(parent_dir,Img_directory)
train_dir = 'Train'
test_dir = 'Test'
train_img = os.path.join(save_img_path,train_dir)
test_img = os.path.join(save_img_path,test_dir)

try:
    os.mkdir(save_img_path)
    os.mkdir(train_img)
    os.mkdir(test_img)

except OSError as error:
    print(error)


faceCascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

while webcam.isOpened():
    
    flag,frame = webcam.read()
    cv.imshow('MainVideo',frame)
    
    # If face is detected and cropped image is returned

    if image_collect(frame) is not None:
        faces = image_collect(frame)
                
        try:
            counter += 1
            img_pil = Image.fromarray(faces)
            img_array = np.array(img_pil)
            cv.putText(frame, str(counter), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            save_image(counter , faces )
            cv.imshow('Face_Cropped', faces)
        
        except Exception as e:
            print(e)        

        
    else:
        print('Face not found')                 
    
    # press 'Q' to quite or webcam is closed after 100 picture is taken

    if cv.waitKey(1) == ord('q') or counter == 100:
        break
  


webcam.release()
cv.destroyAllWindows()