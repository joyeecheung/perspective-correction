#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import time
import argparse

import cv2
import numpy as np

from PIL import Image

from perspective import correct_perspective

SRC_DIR = 'dataset'
DEST_DIR = 'result'

def generate_results(src, dest, temp=True):
    print 'Processing', src + '...'
    im = np.asarray(Image.open(src))
    if (temp):
        gray, blurred, edges, lines, corners, final = correct_perspective(im)
        gray.save(dest % 'gray')
        blurred.save(dest % 'blur')
        edges.save(dest % 'edges')
        lines.save(dest % 'lines')
        corners.save(dest % 'corners')
        final.save(dest % 'final')
        print 'saved', dest
    else:
        correct_perspective(im, temp=temp)

def get_template(filename):
    base, ext = os.path.splitext(filename)
    template = base + '-%s' + ext
    return template

def main():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir, _ = os.path.split(file_dir)
    src_path = os.path.join(parent_dir, SRC_DIR)
    dest_path = os.path.join(parent_dir, DEST_DIR)

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--time", action='store_true')
    args = parser.parse_args()

    print 'Source path: ' + src_path
    if not os.path.exists(src_path):
        raise Exception("Source path doesn't exist!")
    print 'Result directory: ' + dest_path
    if not os.path.exists(dest_path):
        print "Result directory does not exist, created."
        os.makedirs(dest_path)

    filenames = glob.glob(os.path.join(src_path, '*.jpg'))
    
    if args.time:
        start = time.time()
        for name in filenames:
            template = get_template(name.replace(SRC_DIR, DEST_DIR))
            generate_results(name, template, temp=False)
        print "%f seconds wall time" % (time.time() - start)
    else:
        for name in filenames:
            template = get_template(name.replace(SRC_DIR, DEST_DIR))
            generate_results(name, template)


if __name__ == '__main__':
    main()
