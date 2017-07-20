"""Used for the dataset senz3d_datasets
Change raw depth maps (320 x 240 16 bit) to png images
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
from PIL import Image


def find_filenames(root_dir, keywords=None):
    """Look up recursively in root_dir to find all files contating keywords

    Args:
      root_dir: the root directory from where we start
      keywords: find the names containing all of these keywords, by defaut
        None, then we return all file paths

    Returns:
      A list of file paths such that the names contain keywords
    """
    res_filenames = []

    for filename in os.listdir(root_dir):
        path = os.path.join(root_dir, filename)
        if os.path.isdir(path):
            res_filenames.extend(find_filenames(path, keywords))
        elif keywords is None:
            res_filenames.append(path)
        else:
            to_add = True
            for keyword in keywords:
                if keyword not in filename:
                    to_add = False
                    break
            if to_add:
                res_filenames.append(path)
    return res_filenames


def raw_to_png(root_dir, keywords, height=240, width=320):

    filenames = find_filenames(root_dir, keywords)

    for i, filename in enumerate(filenames):
        image_data = np.fromfile(filename, 'int16')
        image_data = image_data.reshape(height, width)
        image_data = (4095-image_data)/4096
        image = Image.fromarray((image_data*255).astype(np.uint8))
        name, _ = os.path.splitext(filename)
        image.save(name + '.png')
        sys.stdout.write('\r>> Converting image %d/%d' % (
                         (i+1), len(filenames)))
        sys.stdout.flush()


def delete_files(root_dir, keywords):

    filenames = find_filenames(root_dir, keywords)

    for i, filename in enumerate(filenames):
        os.remove(filename)
        sys.stdout.write('\r>> Deleting image %d/%d' % (
                         (i+1), len(filenames)))
