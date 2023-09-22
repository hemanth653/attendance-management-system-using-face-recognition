import cv2
import datetime
import myapi as mapi

recog=cv2.face.LBPHFaceRecognizer_create()
recog.read('./facemodel.yml')

names={0:"Hemanth",1:"varsha",2:"pavan",3:"kantesh"}

cam=cv2.VideoCapture(0)
cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:

    flag,frame=cam.read()
    
    # frame=cv2.resize(frame,(100,100))
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces=cascade.detectMultiScale(gray,1.1,5)
    
    if len(faces)>0:
        cropped=""
        
        x,y,w,h=faces[0]
            
        cropped=gray[y:y+h,x:x+w]
        reduced=cv2.resize(cropped,(300,300))
        label,confi=recog.predict(reduced)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),1)
        print(f"Label = {label}, confidence = {confi}")
            
        if confi <=80 :
            detected=names[label]
            current_datetime = datetime.datetime.now()
            mapi.saveData(detected)
            # User-defined format
            user_defined_format = "%d-%B-%Y %H:%M:%S"  # Example format: "2023-08-31 15:30:00"

            # Format the current datetime using the user-defined format
            cdt = current_datetime.strftime(user_defined_format)   #it is used to convert date to string format
            
            with open('students.txt', "a") as file:
                file.write(f"{detected},{cdt}"+ "\n")

                
                
        else:
            detected="Unknown"
            
        cv2.putText(frame,detected,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA)
        
    cv2.imshow("testimage",frame)
    k=cv2.waitKey(3)
    
    if k==ord('q'):
        break

   


cam.release()