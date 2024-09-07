import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import os

from estimate_backscatter import estimate_backscatter
from load_raw import load_image, load_depthmap, preprocess_depthmap
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, required=True, help='Path for Input Images')
    parser.add_argument('--depthmap-path', type=str, required=True, help='Path for Depthmaps corresponding to the Input Images')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save the backscatter removed images')
    args = parser.parse_args()

    dirs = os.listdir(args.image_path)[:100]

    print("Detected {} files in directory {}\n".format(len(dirs), args.image_path))
    times=[]
    for file in tqdm(dirs, desc = 'dirs'):
        start=time.time()
        f = os.path.join(args.image_path, file)
        t = time.time()
        image = load_image(f)
        depths = preprocess_depthmap(image, load_depthmap(os.path.join(args.depthmap_path,f"{file.split('.')[0]}.png"), (image.shape[1], image.shape[0])))

            
        Ba, _ = estimate_backscatter(image, depths)

        Da = image - Ba
        Da = np.clip(Da, 0, 1)
        D = np.uint8(Da * 255.0)
        backscatter_removed = Image.fromarray(D)
        backscatter_removed.save(args.output_path +'/'+file)
        times.append(time.time()-start)
    print(np.mean(times))

    print("Done!")

'''
python preprocess.py --image-path "/media/suhas/Data/uwe_hemang/Datasets/EUVP/raw_inputs" --depthmap-path "/media/suhas/Data/uwe_hemang/Datasets/EUVP/depthmaps" --output-path "/media/suhas/Data/uwe_hemang/Datasets/EUVP/bsr_images"
'''
