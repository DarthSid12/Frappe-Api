import base64

from flask import Flask,request
from flask_restful import  Api, Resource, reqparse
import os
import cv2
import numpy as np
import time
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from google.cloud import firestore as fire
import io
from PIL import Image

app = Flask(__name__)
api = Api(app)

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

firestore_db = firestore.client()

foodObjects = ["apple", "banana", "sandwich", "orange", "broccoli", "carrot", "donut", "cake", "pizza", "hot dog"]

# Define and parse input arg
MODEL_NAME = "coco_ssd_mobilenet_v1"
print("Model Name =",MODEL_NAME)
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = "labelmap.txt"
min_conf_threshold = float(0.5)
resW, resH = "1280x720".split('x')
imW, imH = int(resW), int(resH)
use_TPU = False

print("Use TPU:-",use_TPU)

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
from tensorflow.lite.python.interpreter import Interpreter

    # Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map when using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del (labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

def processImage(base64String):

    try:
        base64String = base64String.decode()
    except:
        pass
    # Conv
    # ''.replace(__old='',__new='')
    base64String = base64String.replace(' ',"+").strip()
    imgdata = base64.b64decode(base64String)
    filename = 'some_image.jpg'  # I assume you have a way of picking unique filenames
    with open(filename, 'wb') as f:
        f.write(imgdata)
    image = Image.open(io.BytesIO(imgdata))
    image.save('TestImage.png')
    frame_rgb = 0
    try:
        frame_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    except  :
        frame_rgb = cv2.cvtColor(np.array(image)[0], cv2.COLOR_BGR2RGB)

    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    # boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
    # num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

    # Loop over all detections and add to recognizedObjects if confidence is above minimum threshold
    recognizedObjects = []
    for i in range(len(scores)):
        if (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()

            # Draw label
            object_name = labels[int(classes[i])]  # Look up object name from "labels" array using class index

            # Print info
            print('Object ' + str(i) + ': ' + object_name)
            recognizedObjects.append({'object': object_name, 'accuracy': scores[i]})

    print(recognizedObjects)

    return recognizedObjects

Parser = reqparse.RequestParser()
Parser.add_argument('image', type=str, help='Image base64 string')
Parser.add_argument('id', type=str, help='The fridge ID')

class AddItem(Resource):
    def post(self):
        # args2 = Parser.parse_args()
        # image = args2['image']
        # print(request.form)
        image = request.form['image']
        print(image)
        # fridge_id = args2['id']
        fridge_id = request.form['id']
        if (request.form['code'] != "superSecretFrappeCodeIs90964532"):
            return {'completed': False, "Reason": "Incorrect verification code"}

        # fridge_id = '2PJvDrpYIUZfQ3DLufj5'
        recognizedObjects = processImage(image)
        outputObjects = []
        print("Recognised objects",recognizedObjects)
        for ob in recognizedObjects:
            if ob['object'] in foodObjects:
                print("Almost here")
                itemReference = firestore_db.collection('Fridges').document(fridge_id).collection('itemInfo').document(
                    ob['object'])
                print("Almost here 2")
                item = itemReference.get()
                print(item.to_dict())
                print("Almost here 3")
                if not item.exists:
                    itemReference.set({
                        'expiryDays': 7,
                        'name': ob['object'],
                        'quantity': 1,
                        'items': [],
                        'requiredQuantity': 5
                    })
                print("Almost here 4")
                itemReference.update({'items':fire.ArrayUnion([int(time.time() * 1000)])})
                print("Almost here 5")
                outputObjects.append(ob['object'])

        output = {'completed':True,'objects':outputObjects}
        return output


class DeleteItem(Resource):
    def post(self):
        image = request.form['image']
        print(image)
        # fridge_id = args2['id']
        fridge_id = request.form['id']
        if (request.form['code']!="superSecretFrappeCodeIs90964532"):
            return {'completed':False,"Reason":"Incorrect verification code"}

        recognizedObjects = processImage(image)
        outputObjects = []
        print("Recognised objects", recognizedObjects)
        for ob in recognizedObjects:
            if ob['object'] in foodObjects:
                itemReference = firestore_db.collection('Fridges').document(fridge_id).collection('itemInfo').document(
                    ob['object'])
                item = itemReference.get()
                try:
                    x = item.to_dict()['items'][0]
                    itemReference.update({'items': fire.ArrayRemove([x])})
                except:
                    pass
                outputObjects.append(ob['object'])
        output = {'completed': True,'objects':outputObjects}
        return output
        # use parser and find the user's query



class Test(Resource):
    def get(self):
        output = {'working':'Yay'}
        return output
# Setup the Api resource routing here
# Route the URL to the resource


api.add_resource(AddItem, '/add')
api.add_resource(DeleteItem, '/delete')
api.add_resource(Test, '/')


if __name__ == '__main__':
    app.run(debug=False)    
