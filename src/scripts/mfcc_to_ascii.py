"""Used for the AVletters dataset
Decode htk mfcc formats to ascii using HList
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import subprocess


def mfcc_to_ascii(source_dir, dest_dir):

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    i = 0
    for filename in os.listdir(source_dir):
        name, ext = os.path.splitext(filename)

        if ext == '.mfcc':

            i += 1
            sys.stdout.write('\r>> Converting mfcc %d' % 1)
            sys.stdout.flush()

            path = os.path.join(source_dir, filename)
            file_content = subprocess.check_output(['HList', path])

            with open(os.path.join(dest_dir, name), 'w') as out_file:
                out_file.write(file_content)
