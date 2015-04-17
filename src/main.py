#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse

import cv2

from files import NamesGetter
from perspective import correct_perspective

SRC_DIR = 'dataset'
DEST_DIR = 'result'
IMG_NAMES = ()

def generate_results(src, dest):
    print 'processing', src + '...'
    im = cv2.imread(src)
    edges, lines, corners, final = correct_perspective(im)
    edges.save(dest % 'edges')
    lines.save(dest % 'lines')
    corners.save(dest % 'corners')
    final.save(dest % 'final')
    print 'saved', dest

def main():
    get_name_pair = NamesGetter(SRC_DIR, DEST_DIR)
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    parser.add_argument("-i", "--input", type=str)

    if args.input is not None:
      src, dest = get_name_pair(args.input)
      generate_results(src, dest)

    else:
      for name in IMG_NAMES:
        src, dest = get_name_pair(name)
        generate_results(src, dest)


if __name__ == '__main__':
    main()