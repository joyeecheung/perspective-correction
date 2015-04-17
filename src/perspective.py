#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import cv2

def get_corners(lines):


def correct_perspective(img):
  # ------------- detect edges ---------------
  edges = ...
  edges_im = Image.fromarray(edges)

  # ------------- get lines ------------------
  lines = ...
  lines_anonated = ...

  # ------------- compute corners ------------
  corners = ...
  corners_annotated = ...

  # ------------- warp ----------------------

  # get trasnformation matrix

  # warp the image
  final = ...

  return (edges_im,
          lines_annotated,
          corners_annotated,
          Image.fromarray(final))