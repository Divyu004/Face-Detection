import cv2

cascade_classifier = cv2.CascadeClassifier('D:/Prg/Python/FD/Face-Detection/haarcascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(1) #1 shows that we are using external webcam To use inbuilt webcam replace 1 with 0

while True:
    
    # Read the frame
    ret, frame = cap.read() # ret indicates whether the frame was read correctly or not
                            # frame is frame of the vid itself
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect Faces
    detections = cascade_classifier.detectMultiScale(gray, 
                                                     scaleFactor = 1.1, 
                                                     minNeighbors =  5,
                                                     minSize = (30,30))

    # Draw the rectangle around each face
    print("Found {0} faces!".format(len(detections)))
    for (x,y,w,h) in detections:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
                                                #   (B,G,R)   Thickness
    # Display the resulting frame
    cv2.imshow('Face Detection', frame)
    key = cv2.waitKey(1)
    if key == ord('x'): #Press 'x' to end the display
        break

cap.release()
cv2.destroyAllWindows()
