import onnx
from onnx import TensorProto
import numpy as np
import coremltools as mlt

#model_path = 'yolov8n-kopia.onnx'
#model = onnx.load(model_path)
#
#graph = model.graph
#
## mlt.converters.mil.mil.ops.defs.iOS17.tensor_transformation.slice_by_index(
#
##transpose_scores_node = onnx.helper.make_node(
##    'Transpose',
##    inputs=['output0'],
##    outputs=['output0_transposed'],
##    perm=(0, 2, 1))
##    
##graph.node.append(transpose_scores_node)
#
#max_detections = 200
#score_thresh = 0.25
#iou_thresh = 0.5
#
#score_threshold = onnx.helper.make_tensor(
#    'score_threshold',
#    TensorProto.FLOAT,
#    [1],
#    [score_thresh])
#
#iou_threshold = onnx.helper.make_tensor(
#    'iou_threshold',
#    TensorProto.FLOAT,
#    [1],
#    [iou_thresh])
#
#max_output_boxes_per_class = onnx.helper.make_tensor(
#    'max_output_boxes_per_class',
#    TensorProto.INT64,
#    [1],
#    [max_detections])
#    
#inputs_nms=['iou_threshold', 'score_threshold']
#outputs_nms = ['num_selected_indices']
#
#nms_node = onnx.helper.make_node(
#    'NonMaxSuppression',
#    inputs_nms,
#    outputs_nms,
#    center_point_box=1,
#)
#
## add to the list of graph nodes
#graph.node.append(nms_node)
#
## initializer
#graph.initializer.append(score_threshold)
#graph.initializer.append(iou_threshold)
#graph.initializer.append(max_output_boxes_per_class)
#
## define output
#output_nms_value_info = onnx.helper.make_tensor_value_info(
#    'num_selected_indices',
#    TensorProto.INT64,
#    shape=['num_selected_indices', 3])
#
## add to graph
#graph.output.append(output_nms_value_info)
#
#
#onnx.save(model, 'modified-model.onnx')


## ADDING NMS

# * model is the YOLOv8n trained (YOLO class)
# * batch = 1 is important here,
#     even export support dynamic axis using dynamic=True,
#     sometimes it just fail to export
#model.export(format='onnx', simplify=True, imgsz=[640,640], batch=1)

## load the model and manipulate it
#model_path = 'yolov8n-kopia.onnx'
#onnx_model = onnx.load(model_path)
##onnx_fpath = f"{weight_folder}/best_nms.onnx"
#
#graph = onnx_model.graph
#
## operation to transpose bbox before pass to NMS node
#transpose_bboxes_node = onnx.helper.make_node("Transpose",inputs=["/model.22/Mul_2_output_0"],outputs=["bboxes"],perm=(0,2,1))
#graph.node.append(transpose_bboxes_node)
#
## make constant tensors for nms
#score_threshold = onnx.helper.make_tensor("score_threshold", TensorProto.FLOAT, [1], [0.25])
#iou_threshold = onnx.helper.make_tensor("iou_threshold", TensorProto.FLOAT, [1], [0.45])
#max_output_boxes_per_class = onnx.helper.make_tensor("max_output_boxes_per_class", TensorProto.INT64, [1], [200])
#
## create the NMS node
#inputs=['bboxes', '/model.22/Sigmoid_output_0', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold',]
## inputs=['onnx::Concat_458', 'onnx::Concat_459', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold',]
#outputs = ["selected_indices"]
#nms_node = onnx.helper.make_node(
#    'NonMaxSuppression',
#    inputs,
#    ["selected_indices"],
#    # center_point_box=1 is very important, PyTorch model's output is
#    #  [x_center, y_center, width, height], but default NMS expect
#    #  [x_min, y_min, x_max, y_max]
#    center_point_box=1,
#)
#
## add NMS node to the list of graph nodes
#graph.node.append(nms_node)
#
## append to the output (now the outputs would be scores, bboxes, selected_indices)
#output_value_info = onnx.helper.make_tensor_value_info("selected_indices", TensorProto.INT64, shape=["num_results",3])
#graph.output.append(output_value_info)
#
## add to initializers - without this, onnx will not know where these came from, and complain that
## they're neither outputs of other nodes, nor inputs. As initializers, however, they are treated
## as constants needed for the NMS op
#graph.initializer.append(score_threshold)
#graph.initializer.append(iou_threshold)
#graph.initializer.append(max_output_boxes_per_class)
#
## remove the unused concat node
#last_concat_node = [node for node in onnx_model.graph.node if node.name == "/model.22/Concat_5"][0]
#onnx_model.graph.node.remove(last_concat_node)
#
## remove the original output0
#output0 = [o for o in onnx_model.graph.output if o.name == "output0"][0]
#onnx_model.graph.output.remove(output0)
#
## output keep for downstream task
#graph.output.append([v for v in onnx_model.graph.value_info if v.name=="/model.22/Mul_2_output_0"][0])
#graph.output.append([v for v in onnx_model.graph.value_info if v.name=="/model.22/Sigmoid_output_0"][0])
#
## check that it works and re-save
#onnx.checker.check_model(onnx_model)
#onnx.save(onnx_model, 'modified-model.onnx')





model_path = 'yolov8n-kopia.onnx'
onnx_model = onnx.load(model_path)
#onnx_fpath = f"{weight_folder}/best_nms.onnx"

graph = onnx_model.graph

reshape_node = onnx.helper.make_node("Squeeze", inputs=["output0"], outputs=["squeezed_output0"])
graph.node.append(reshape_node)

transpose_node = onnx.helper.make_node("Transpose", inputs=["squeezed_output0"], outputs=["final_output0"], perm=(1,0))
graph.node.append(transpose_node)

output_value_info = onnx.helper.make_tensor_value_info("final_output0", TensorProto.FLOAT, shape=[6300,84])
graph.output.append(output_value_info)

# operation to transpose bbox before pass to NMS node
#transpose_bboxes_node = onnx.helper.make_node("Transpose",inputs=["/model.22/Mul_2_output_0"],outputs=["bboxes"],perm=(0,2,1))
#graph.node.append(transpose_bboxes_node)


onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, 'modified-model.onnx')
