import onnx
import pprint


pprint.pprint([protocol for protocol in dir(onnx) 
               if protocol.endswith('Proto') and protocol[0] != '_'])