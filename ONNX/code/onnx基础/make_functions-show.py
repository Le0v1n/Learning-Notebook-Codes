import onnx
import pprint


pprint.pprint([k for k in dir(onnx.helper) if k.startswith('make')])