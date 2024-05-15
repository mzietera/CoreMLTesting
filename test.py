import onnxmltools
import coremltools

# Load a Core ML model
coreml_model = coremltools.utils.load_spec('yolov8n.mlpackage')

# Convert the Core ML model into ONNX
onnx_model = onnxmltools.convert_coreml(coreml_model, 'Example Model')

# Save as protobuf
onnxmltools.utils.save_model(onnx_model, 'example.onnx')
