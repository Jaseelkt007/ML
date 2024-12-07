import numpy as np
from PIL import Image
from multiprocessing import Pool
import warnings
from tqdm import tqdm
import os
import cv2

def trim(image):

    percentage = 0.02
    img = np.array(image)
    img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) # Convert to grayscale to simply the process
    # create the binary mask , to get the background from actual content
    img_gray = img_gray > 0.1 * np.mean(img_gray[img_gray!=0])
    # calculate the row wise and column wise sums to find where the significant content exists
    row_sums = np.sum(img_gray, axis = 1)
    col_sums = np.sum(img_gray, axis = 0)
    rows = np.where(row_sums > img.shape[1] * percentage)[0] # return the rows index of rows which contain at least 2% of its content
    cols = np.where (col_sums > img.shape[0] * percentage)[0]
    # find the min and max rows and columns for croping
    min_row, min_col = np.min(rows), np.min(cols)
    max_row, max_col = np.max(rows), np.max(cols)
    im_crop = img[min_row : max_row +1 , min_col : max_col+1]
    return Image.fromarray(im_crop)

def resize_main_aspect(image, desired_size):
    old_size = image.size
    ratio = float(desired_size)/ max(old_size) # resize ratio
    new_size = tuple([int(x * ratio) for x in old_size]) # (N,M) N,M are new size
    im = image.resize(new_size, Image.LANCZOS) # a filter to smooth image when resize, helps to reduce artifacts in the reduced image
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0])//2 , (desired_size - new_size[1])//2)) # paster the image on the new square background
    return new_im

def save_single(args): # helpfull for multiprocessing
    img_file, input_path_folder, output_path_folder, output_size = args
    image_org = Image.open(os.path.join(input_path_folder, img_file))
    image = trim(image_org)
    image = resize_main_aspect(image, desired_size= output_size[0])
    image.save(os.path.join(output_path_folder , img_file))



def multi_image_resize(input_path_folder, output_path_folder, output_size=None):
    if not output_size:
        warnings.warn("Need to specify output_size! For example: output_size=100")
        exit()

    if not os.path.exists(output_path_folder):
        os.makedirs(output_path_folder)

    jobs = [
        (file, input_path_folder, output_path_folder, output_size)
        for file in os.listdir(input_path_folder)
        if os.path.isfile(os.path.join(input_path_folder,file))
    ]

    with Pool() as p:
        list(tqdm(p.imap_unordered(save_single, jobs), total=len(jobs)))

if __name__ == '__main__':
    data_paths = [
        {
            "sample_data_path": "/home/data/IDRID_dataset/images/train",
            "output_folder": "/home/RUS_CIP/st186731/revized_images/train",
        },
        {
            "sample_data_path": "/home/data/IDRID_dataset/images/test",
            "output_folder": "/home/RUS_CIP/st186731/revized_images/test",
        }
    ]
    for paths in data_paths:

        if not os.path.exists(paths['output_folder']):
            os.makedirs(paths['output_folder'])

        multi_image_resize(input_path_folder = paths['sample_data_path'], output_path_folder = paths['output_folder'], output_size = (256,256))