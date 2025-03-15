import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--orig-root', type=str, required=True, help='Path for Input Videos')
parser.add_argument('--ffn-root', type=str, required=True, help='Path to FastFlowNet')
parser.add_argument('--flow-root', type=str, required=True, help='Path to save the Backward Optical Flows')
args = parser.parse_args()


orig_root = args.orig_root
flow_root = args.flow_root
os.makedirs(flow_root , exist_ok = True)
os.makedirs(f"{flow_root}/forward_flow" , exist_ok = True)
os.makedirs(f"{flow_root}/backward_flow" , exist_ok = True)


ffn_root = args.ffn_root
for dir in ["backward","forward"]:
    for video in os.listdir(f'{orig_root}'):
        in_dir = f'{orig_root}/{video}'
        high_flow_dir = f'{flow_root}/{dir}_flow/{video}/high_flow'
        low_flow_dir = f'{flow_root}/{dir}_flow/{video}/low_flow'
        os.makedirs(f"{flow_root}/{dir}_flow/{video}" , exist_ok =True)
        os.makedirs(high_flow_dir , exist_ok=True)
        os.makedirs(low_flow_dir , exist_ok=True)
        print(f'Started getting {dir} flow for {video}...')
        command = f'python {run_file}/run_{dir}.py --image-path {in_dir} --hflow-path {high_flow_dir} --lflow-path {low_flow_dir}'
        os.system(command)
