#importing the modules
import cv2

#Loading cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
#Creating a function to detect faces and respective eyes
def detect(gray_image, color_image):
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(color_image, (x,y) , (x+w,y+h) , (204,155,0) , 2)
        roi_gray = gray_image[y:y+h,x:x+w]
        roi_color = color_image[y:y+h,x:x+h]
        eyes = eyes_cascade.detectMultiScale(roi_gray, 1.1, 22)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey), (ex+ew,ey+eh), (0,255,0) , 2)
        smile = smile_cascade.detectMultiScale(roi_gray, 1.6 , 22)
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_color,(sx,sy), (sx+sw,sy+sh), (0,0,255),2)
    return color_image

#Now playing with the webcam

video_capture = cv2.VideoCapture(0)
while True:
    _, images= video_capture.read()
    images = cv2.flip(images,1)
    gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    returned_image = detect(gray, images)
    cv2.imshow('Video', returned_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
