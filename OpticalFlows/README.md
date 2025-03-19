# Optical Flows
Use [Fast Flow Net](https://github.com/ltkong218/FastFlowNet) to compute Optical Flows.
Move the files ```run_forward.py and run_backward.py``` to ```FastFlowNet/```
Run ```OpticalFlows/get_flows.py```
```bash
python get_flows.py --orig-root <Train Data> --ffn-root <FastFlowNet> --flow-root <Output Flows>
```
<br><br>
For a help Menu
use ```python <filename> -h```
