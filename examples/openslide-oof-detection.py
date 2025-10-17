"""
TODO: Remove this from examples folder or make it a real slidescore example.
"""
import argparse
import time
import os
import random

import openslide
import skimage
from skimage import io
from skimage.color import rgb2gray
import numpy as np # $ pip install numpy

def apply_mask(orig_img, mask, mask_color):
    overlay = np.zeros_like(orig_img, dtype=np.uint8)

    # Replace the pixels inside the mask with the overlay color
    overlay[mask] = mask_color
    img_overlayed = np.ubyte(0.7 * orig_img + 0.3 * overlay)
    return img_overlayed


def getSharpness(img, ksize=5, background_threshold = 0.9):
    img_gray = rgb2gray(img)
    img_foreground_mask = img_gray < background_threshold
    img_laplace = skimage.filters.laplace(img_gray, ksize=ksize, mask=img_foreground_mask)

    # print(img_fn, "laplace var", np.var(img_laplace) * 10000)    
    sharpness_measure = np.var(img_laplace) * 10000
    return sharpness_measure

def find_tissue_coords(slide: openslide.OpenSlide | openslide.ImageSlide, background_threshold = 0.9, num_samples = 10):
    # First detect tissue by getting a thumbnail
    thumbnail = slide.get_thumbnail((1024, 1024)).convert('RGB')
    thumbnail_img = np.array(thumbnail)
    thumbnail_gray = rgb2gray(thumbnail_img)
    
    # To remove noise of small stuff, blur and then reapply filter to find tissue
    img_foreground_mask = thumbnail_gray < background_threshold
    blurred_foreground = skimage.filters.gaussian(img_foreground_mask, sigma=10)
    tissue_mask = blurred_foreground > background_threshold
    img_overlayed = apply_mask(thumbnail_img, tissue_mask, (255, 255, 0))
    io.imsave("tissue_mask.png", img_overlayed, check_contrast=False)
    # Get all coordinates of the white pixels, which are supposed to be tissue
    tissue_coords = np.transpose(tissue_mask.nonzero())
    if tissue_coords.shape[0] == 0:
        raise Exception("Was not able to detect tissue.")

    # Pick n pixels at random, which will be used as tile offsets eventually, these are (y, x)
    np.random.seed(seed=42)
    random_coords = tissue_coords[np.random.choice(tissue_coords.shape[0], num_samples, replace=False)]
    # Convert these coords to yx ratios, so they are dimension agnostic
    random_coord_ratios = random_coords / thumbnail_gray.shape
    # Return xy ratios
    return np.flip(random_coord_ratios)


def get_slide_tiles(slide: openslide.OpenSlide | openslide.ImageSlide, zoom_level: int, coord_ratios: np.ndarray):
    offsets = coord_ratios * slide.dimensions # Offsets are based on the most zoomed in level

    # print(slide.level_dimensions, zoom_level)
    tile_size = 512
    
    imgs = []
    for i, (x_offset, y_offset) in enumerate(offsets):
        x_offset, y_offset = int(x_offset), int(y_offset)
        img = slide.read_region((x_offset, y_offset), zoom_level, (tile_size, tile_size)).convert('RGB')
        output_dir = 'output'
        fn = f'{output_dir}/tile_x{coord_ratios[i][0]}_y{coord_ratios[i][1]}_z{zoom_level}.png'

        if random.random() < 0.5 and False:
            # Apply a gaussian blur for testing
            # print("blurring", fn)
            img_data = np.array(img)
            img_data_blurred = skimage.filters.gaussian(img_data, sigma=2, channel_axis=-1)
            img = openslide.Image.fromarray(np.uint8(img_data_blurred * 256)).convert('RGB')
        imgs.append((fn, np.array(img)))

        if os.path.isdir(output_dir):
            img.save(fn)
    return imgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='SlideScore openslide OOF detector',
                    description='Takes reports the ratio blurred for a fully zoomed in PNG')
    parser.add_argument('--threshold', type=int, default=10, description='Threshold sharpness value to consider a tile to be blurry')
    parser.add_argument('--background-threshold', type=float, default=0.9)
    parser.add_argument('--file', type=str, required=True)
    args = parser.parse_args()
    t0 = time.time()
    print("{:.2f}".format(time.time() - t0), "Initialized")

    slide = openslide.open_slide(args.file)
    print("{:.2f}".format(time.time() - t0), "Opened slide")
    
    # First detect tissue by getting a thumbnail and extracting dark regions
    tissue_coords = find_tissue_coords(slide, num_samples = 10)
    print("{:.2f}".format(time.time() - t0), "Identified 10 locations to check sharpness")

    # Then extract the slide tiles at these coords
    level = 1 if slide.level_count > 1 else 0
    slide_tiles = get_slide_tiles(slide, level, tissue_coords)
    print("{:.2f}".format(time.time() - t0), "Retrieved 10 images from slide")

    sharpness_vals = []
    num_blurry = 0
    for (fn, img) in slide_tiles:
        sharpness_val = getSharpness(img, background_threshold=args.background_threshold)
        sharpness_vals.append(sharpness_val)

        is_blurry = sharpness_val < args.threshold
        if is_blurry:
            num_blurry += 1
    avg_sharpness = sum(sharpness_vals) / len(sharpness_vals)

    print("{:.2f}".format(time.time() - t0), f"Identified {num_blurry}/10 to be blurry,", "{:.2f}".format(avg_sharpness), "is the avg. sharpness")