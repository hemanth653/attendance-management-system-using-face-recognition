import cv2 as cv
import random

cam = cv.VideoCapture(0)

crop=""

cascade=cv.CascadeClassifier('haarcascade_frontalface_default.xml')  #ia a harr cascade designed by openCV to detect the frontal face.Its works on trainning the cascade on thousands of negativ images with positive images superimpose don it

while True:

    flag,image=cam.read()   #reading the function

    gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)   #the image is converted into the black and white

    faces=cascade.detectMultiScale(gray,1.1,5)  #(image,scale factor, neighbours)
    print(faces)

    for x,y,w,h in faces:
        cv.rectangle(image,(x,y),(x+w,y+h),(25,200,100),2)  #(image,star-point,end-point,color,thickness)
        crop=image[y:y+h,x:x+w]


    cv.imshow("my image",image)  #image is showed in the  other page
    k= cv.waitKey(1)   #which allow the user to display window for given milliseconds or untill any key are present

    if k==ord('h'):
        break
    elif k==ord('s'):
        index=random.randint(1,1000)
        filename=f"./dataset/0/cropped{index}.jpg"
        cv.imwrite(filename,crop)
        
cam.release()