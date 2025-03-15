import rawpy
from PIL import Image
import numpy as np
import os
import argparse

from estimate_backscatter import estimate_backscatter

def load_raw_image(image_path, max_side = 1024):
    """
      This function loads the RAW image with the max(width, height) set to max_side.
    """
    image_file_raw = rawpy.imread(image_path).postprocess()
    image_file = Image.fromarray(image_file_raw)
    image_file.thumbnail((max_side, max_side), Image.Resampling.LANCZOS) #Remove Resampling on newer versions of Pillow
    image_file.save("raw/out/" + image_path[-12:-4] + ".png")
    return np.float64(image_file) / 255.0

def load_image(image_path, max_side = 1024):
    image_file = Image.open(image_path)
    image_file.thumbnail((max_side, max_side), Image.Resampling.LANCZOS) #Remove Resampling on newer versions of Pillow
    return np.float64(image_file) / 255.0

def load_image_metrics(image_path, max_side = 1024):
    image_file = Image.open(image_path)
    image_file.thumbnail((max_side, max_side), Image.Resampling.LANCZOS) #Remove Resampling on newer versions of Pillow
    return np.uint8(image_file)


def load_depthmap(depthmap_path, size):
    depth_file = Image.open(depthmap_path)
    if len(np.array(depth_file).shape) >2:
        depth_file = depth_file.convert("L")
    depths = depth_file.resize(size, Image.Resampling.LANCZOS) #Remove Resampling on newer versions of Pillow
    # depths = np.max(depths) / depths # disparity map to depth map
    return np.float64(depths)

def estimate_far(image, frac = 0.2, close = 0.3):
    r = image[:, :, 0] * 0.2126
    g = image[:, :, 1] * 0.7152
    b = image[:, :, 2] * 0.0722

    lum = np.sum(np.stack([r, g, b], axis = 2), axis = 2)
    lum.sort(axis = 0)
    rows = int(frac * lum.shape[0])

    darkest = np.mean(lum[rows:(2 * rows), :], axis = 0)
    brightest = np.mean(lum[-(2 * rows):(-rows), :], axis = 0)

    # ratio = np.mean(brightest / darkest)
    ratio = np.mean(np.divide(brightest, darkest, where=darkest!=0), where=darkest!=0)

    return np.log2(ratio) * 10

def preprocess_depthmap(image, depths):
    # print(np.min(image), np.max(image), np.min(depths), np.max(depths))
    far = estimate_far(image)
    # print("Estimated farthest distance is {far}".format(far = far))
    # ratio = far / np.max(depths)
    # print(far, np.max(depths), np.iinfo('uint16').max)
    ratio = far / (np.max(depths)/np.iinfo('uint16').max)
    # print(ratio)

    # print("Estimated mean distance is {mean}".format(mean = np.mean(depths * ratio)))
    return depths * ratio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--remove-backscatter', type=bool, default=False, help='Specify whether to remove backscatter too. If yes, depth map is also needed')
    args = parser.parse_args()

    directory = 'raw/in/'
    print("Detected {} files in directory {}\n".format(len(os.listdir(directory)), directory))
    for file in os.listdir(directory):
        f = os.path.join(directory, file)
        if f[-3:] == "ARW":
            print("Processing file {}".format(file))
            image = load_raw_image(f)
            if args.remove_backscatter is True:
                depths = preprocess_depthmap(image, load_depthmap(os.path.join(directory, file[:-3] + "png"), (image.shape[1], image.shape[0])))

                Ba, _ = estimate_backscatter(image, depths)

                Da = image - Ba
                Da = np.clip(Da, 0, 1)
                D = np.uint8(Da * 255.0)
                backscatter_removed = Image.fromarray(D)
                backscatter_removed.save("raw/out/" + file[:-4] + "_BSR.png")
