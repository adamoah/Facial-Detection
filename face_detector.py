import cv2 as cv
import numpy as np
import argparse

#detects if any faces exist within the current frame and displays the frame with a red box over any faces present
def detectAndDisplay(frame):

    #convert frame to grayscale and increase image contrast
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    grayFrame = cv.equalizeHist(grayFrame)

    #list of all detected faces as rectangles within the current frame
    faces = faceCascade.detectMultiScale(grayFrame)

    #highlight a red rectangular outline around any faces detected in the frame and override the current frame
    #with the red box outline
    for (x,y,w,h) in faces:
        topLeft = (x, y)
        bottomRight = (x + w, y + h)
        frame = cv.rectangle(frame, topLeft, bottomRight, (0, 0, 255), 2)
    
    #display the modified frame
    cv.imshow("Face detector", frame)

#parse argument for reading in face detection model
parser = argparse.ArgumentParser(description="Code for Face Cascade")
parser.add_argument('--faceCascade', help='Path to face cascade', default='haarcascade_frontalface_alt.xml')
parser.add_argument('--camera', help='Camera divide numer', type=int, default=0)
args = parser.parse_args()

#initialize Cascade of facial feature classifiers
faceCascadeName = args.faceCascade
faceCascade = cv.CascadeClassifier()

#loads in sample model data
cv.samples.addSamplesDataSearchPath(r'C:\Users\Golde\source\repos\Computer_Vision\Computer_Vision\env\Lib\site-packages\cv2\data')

#attempts to load sample data into the cascade
if not faceCascade.load(cv.samples.findFile(faceCascadeName)):
    print("Could not load face cascade exiting...")
    exit()

#initialize video capture from live camera feed
cap = cv.VideoCapture(0)
run = True

#apply facial detection function to each frame of the live feed
while run:

    #read in the current frame
    ret, frame = cap.read()

    #exit if some error occurs reading in the frame
    if frame is None:
        print("Could not grab frame exiting capture stream")
        exit()

    #run current frame through the function (will display the frame with a red outline if any faces are present)
    detectAndDisplay(frame)

    #end video capture if user press the q key
    run = False if cv.waitKey(5) == ord('q') else True

#close the capture and any open output windows
cap.release()
cv.destroyAllWindows()
