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
RESULTS = ['gray', 'blur', 'edges', 'lines', 'corners', 'final']

def generate_results(src, dest, time=False,
                     resize=False, intermediate=True):
    print 'Processing', src + '...'
    im = Image.open(src)
    img = np.asarray(im)

    if time:
        correct_perspective(img, intermediate=False)
    elif intermediate:
        results = correct_perspective(img)
        for idx, result in enumerate(results):
            if resize:
                newsize = tuple(int(i/3.0) for i in im.size)
                result.resize(newsize, Image.ANTIALIAS).save(dest % RESULTS[idx])
            else:
                result.save(dest % RESULTS[idx])
            print 'saved', dest % RESULTS[idx]
    else:
        final = correct_perspective(img, intermediate=intermediate)[-1]
        if resize:
            newsize = tuple(int(i/3.0) for i in im.size)
            final.resize(newsize, Image.ANTIALIAS).save(dest % 'final')
        else:
            final.save(dest % 'final')
        print 'saved', dest % 'final'
            

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
    parser.add_argument("-r", "--resize", action='store_true')
    parser.add_argument("-i", "--intermediate", action='store_true')
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
            generate_results(name, template, time=True)
        print "%f seconds wall time" % (time.time() - start)
    else:
        for name in filenames:
            template = get_template(name.replace(SRC_DIR, DEST_DIR))
            generate_results(name, template,
                             intermediate=args.intermediate,
                             resize=args.resize)

if __name__ == '__main__':
    main()
