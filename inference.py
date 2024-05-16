import cv2
import sys
import json
import onnx
import numpy as np
import onnxruntime
from onnx import numpy_helper

def flatten_list(nested_list):
    flat_list = []
    for sublist in nested_list:
        for item in sublist:
            flat_list.append(item)
    return flat_list

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
    prob = row[5:].max()
    class_id = row[5:].argmax()
    label = yolo_classes[class_id]
    return [x1,y1,x2,y2,label,prob]

model_dir ="./"
model=model_dir+"yolov5n.onnx"
path=sys.argv[1]

img = cv2.imread(path)
# img = cv2.resize(img, (640,640), interpolation = cv2.INTER_AREA)
#print(img)
#input = np.array(img)
#print(input.shape)
#img.resize((640, 640, 3))
input = np.array(img)
print(input.shape)

#print(len(input))
#print(len(input[0]))
#print(len(input[0][0]))
#print(input[1][1][0])
#print(input[2][2][1])
#print(input[3][3][2])
print(input.shape)
input = input.transpose(2,0,1)
print(input.shape)
input = input.reshape(1,3,640,640)
print(input[0][0][0][:640])
imageArray = input[0][0][0]
np.set_printoptions(threshold=sys.maxsize)
print(imageArray)
# flattened = flatten_list(imageArray)
# with open("output.txt", "w") as f:
#   content = str(imageArray)
#   f.write(content)

input = (input/255.0).astype(np.float32)


print(len(input[0][1]))
print(input[0][0][0][:640])

# for i in range(640):
#   print(input[0][0][i])
#data = json.dumps({'data': img.tolist()})
#data = np.array(json.loads(data)['data']).astype('float32')
session = onnxruntime.InferenceSession(model, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(input_name)
print(output_name)

result = session.run([output_name], {input_name: input})

print(len(result))
print(len(result[0]))
print(len(result[0][0]))
print(len(result[0][0][0]))

#prediction=int(np.argmax(np.array(result).squeeze(), axis = 0))
#print(prediction)
#print(len(result[0][0].squeeze()))
boxes = [row for row in [parse_row(row) for row in result[0][0]] if (row[5]>0.9 and row[4]=="person")]
#print(len(boxes))
print(len(boxes))
print(boxes)
#for x in range(0, 6299):
#  for y in range(x*84+4, x*84+4+79):
#    print(result[x][y])
#print(result[0][0].size)
#print(result[0][0])


