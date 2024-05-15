import cv2
import sys
import json
import onnx
import numpy as np
import onnxruntime
from onnx import numpy_helper


yolo_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def parse_row(row):
    xc,yc,w,h = row[:4]
    x1 = (xc-w/2)
    y1 = (yc-h/2)
    x2 = (xc+w/2)
    y2 = (yc+h/2)
    prob = row[4:].max()
    class_id = row[4:].argmax()
    label = yolo_classes[class_id]
    return [x1,y1,x2,y2,label,prob]

model_dir ="./"
model=model_dir+"/modified-model.onnx"
path=sys.argv[1]

img = cv2.imread(path)
print(img)
#img.resize((1, 3, 640, 480))
input = np.array(img)
print(len(input))
print(len(input[0]))
print(len(input[0][0]))
#print(input[1][1][0])
#print(input[2][2][1])
#print(input[3][3][2])
print(input.shape)
input = input.transpose(2,0,1)
print(input.shape)
input = input.reshape(1,3,640,480)
input = (input/255.0).astype(np.float32)
print(input)
#data = json.dumps({'data': img.tolist()})
#data = np.array(json.loads(data)['data']).astype('float32')
session = onnxruntime.InferenceSession(model, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[1].name

print(input_name)
print(output_name)

result = session.run([output_name], {input_name: input})
#prediction=int(np.argmax(np.array(result).squeeze(), axis = 0))
#print(prediction)
#print(len(result[0][0].squeeze()))
boxes = [row for row in [parse_row(row) for row in result[0]] if row[5]>0.9]
print(len(boxes))
print(boxes)
#for x in range(0, 6299):
#  for y in range(x*84+4, x*84+4+79):
#    print(result[x][y])
#print(result[0][0].size)
#print(result[0][0])


