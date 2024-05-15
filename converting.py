import coremltools
import coremltools.proto.FeatureTypes_pb2 as ft

#coremlmodel = ct.utils.load_spec("rtmpose-m.mlpackage")
#image_input = ct.converters.mil.input_types.ImageType(name='input_image', shape=(256, 192, 3))
#mlmodel_image = ct.convert(coremlmodel, inputs=[image_input])
coremltools.converters.onnx.convert()
spec = coremltools.utils.load_spec("rtmpose-m.mlpackage")
input = spec.description.input[0]
input.type.imageType.colorSpace = ft.ImageFeatureType.RGB
input.type.imageType.height = 112
input.type.imageType.width = 112
coremltools.utils.save_spec(spec, "YourNewModel.mlpackage")
