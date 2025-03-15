import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video-path', type=str, required=True, help='Path for Input Videos')
parser.add_argument('--depthmap-path', type=str, required=True, help='Path for Depthmaps corresponding to the Input Videos')
parser.add_argument('--output-path', type=str, required=True, help='Path to save the backscatter removed Frames')
args = parser.parse_args()

orig_root = args.video_path
bsr_root = args.depthmap_path
depthmaps = args.output_path
os.makedirs(bsr_root , exist_ok = True)
run_file = 'bsr_per_video.py'
for video in os.listdir(f'{orig_root}'):
    in_dir = f'{orig_root}/{video}'
    depth_dir = f'{depthmaps}/{video}'
    out_dir = f'{bsr_root}/{video}'
    os.makedirs(f'{bsr_root}/{video}' , exist_ok=True)
    print(f'Started getting depthmaps for {video}...')
    command = f'python "{run_file}" --image-path "{in_dir}" --depthmap-path "{depth_dir}" --output-path "{out_dir}/"'
    os.system(command)


'''
python bsr.py --video-path "" --depthmap-path "" --output-path ""
'''