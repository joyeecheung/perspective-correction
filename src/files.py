#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

from PIL import Image

def NamesGetter(src_dir, dest_dir):
    """Return list of tuples for source and template destination
       filenames(absolute filepath)."""
    file_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir, _ = os.path.split(file_dir)
    src_path = os.path.join(parent_dir, src_dir)
    dest_path = os.path.join(parent_dir, dest_dir)

    print 'Source path: ' + source_path
    if not os.path.exists(source_path):
        raise Exception("Source path doesn't exist!")
    print 'Result directory: ' + dest_dir
    if not os.path.exists(dest_dir):
        print "Result directory does not exist, created."
        os.makedirs(dest_dir)

    def get_name_pair(filename):
        base, ext = os.path.splitext(filename)
        tempname = base + '-%s' + ext
        return (os.path.join(src_path, filename),
                          os.path.join(dest_path, tempname))

    return get_name_pair
