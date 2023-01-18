import cv2
from blinkpy.blinkpy import Blink
from blinkpy.blinkpy import Auth
from blinkpy.helpers.util import json_load
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
import urllib.request, urllib.parse
import requests
from datetime import datetime
import time
import sys
import os

def sendNoti(): # sending notification
    r = requests.post("https://api.pushover.net/1/messages.json", data = {
        'token': '',
        'user': '',
        'message': 'Possible Threat. Click to View Image.'
    },
    files = {
        'attachment': ('image.jpg', open('image.jpg', 'rb'), 'image/jpeg')
    })
    

def activateSession():
    blink = Blink()
    blink.auth = Auth(json_load('./credentials.json'))
    blink.start()
    return blink

def getNewClip(camera, file): #checks for new clip
    try:
        camera.video_to_file(file)
    except:
        print("No Video.")
        return False
    return True

def detectObject(file, detector):
    seconds = 500
    probabilityCount=0
    probability = 0
    cap = cv2.VideoCapture(file)
    cap.set(0, seconds)
    success, image = cap.read()
    if success:
        cv2.imwrite('image.jpg', image)

    while cap.isOpened():
        
        cap.set(0, seconds)
        success, image = cap.read()
  
        if success:
            
            image = cv2.flip(image, 1)
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # get frame flipped and inverted
            
            input_tensor = vision.TensorImage.create_from_array(rgb_image)
    
            detection_result = detector.detect(input_tensor)

            image = utils.visualize(image, detection_result)
            
            if cv2.waitKey(1) == 27:
              break
            cv2.imshow('object_detector', image)
          
            for detection in detection_result.detections: # calculate results
                #print(detection)
                category = detection.categories[0]
                category_name = category.category_name
                probability += round(category.score, 2)
                probabilityCount += 1
            seconds += 500
        else:
            if probabilityCount != 0:
                probability /= probabilityCount
            #print("Frame Count:" + str(probabilityCount))
            print("Probability:" + str(probability))
            if probability >= 0.50 or probabilityCount == 0:
                cap.release()
                cv2.destroyAllWindows()
                return False # not a threat
            else:
                cap.release()
                cv2.destroyAllWindows()
                return True # is a threat
    cap.release()
    cv2.destroyAllWindows()
def main():
    # initialize detector options
    counter = 0
    base_options = core.BaseOptions(file_name='model.tflite', num_threads=3)
    detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.3)
    options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)
    file = 'newClip.mp4'
    
    blink = activateSession()
    done = False
    blink.refresh(force=True)
    garage = blink.cameras['Garage']
    
    while done == False: # done == True is never reached
        if getNewClip(garage, file) == False:
            time.sleep(60)
            blink.refresh()
            garage = blink.cameras['Garage']
        else:
            isThreat = detectObject(file, detector)
            if isThreat:
                print("["+datetime.now().strftime("%H:%M:%S")+"]: Notify")
                sendNoti()
            time.sleep(5)
            blink.start()
            blink.refresh()
            garage = blink.cameras['Garage']
    
if __name__ == "__main__":
    main()
    
