# Optical Flows
Use [Fast Flow Net](https://github.com/ltkong218/FastFlowNet) to compute Optical Flows.
Move the files ```./OpticalFlows/run_forward.py and ./OpticalFlows/run_backward.py``` to ```FastFlowNet/```
Run ```OpticalFlows/get_flows.py```
```bash
python OpticalFlows/get_flows.py --orig-root <Train Data> --ffn-root <FastFlowNet> --flow-root <Output Flows>
```
