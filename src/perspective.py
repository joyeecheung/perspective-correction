#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import cv2

def get_corners(lines):
  return 

def annotate_lines(img, lines):
  annnotated = np.array(img)
  width, height, _ = annnotated.shape
  for rho, theta in lines[0]:
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    x1, y1 = int(x0 + width * (-b)), int(y0 + width * (a))
    x2, y2 = int(x0 - height * (-b)), int(y0 - height * (a))
    cv2.line(annnotated, (x1, y1), (x2, y2), (0, 0, 255), 10)

  return annnotated.astype(np.uint8)

def correct_perspective(img, threshold_max=200,
                        threshold_min=100,
                        rho=1,
                        theta=np.pi/180,
                        threshold_intersect=400):
  # ------------- get binary image -----------
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray_im = Image.fromarray(gray)

  # ------------- detect edges ---------------
  edges = cv2.Canny(gray, threshold_min, threshold_max)
  edges_im = Image.fromarray(edges)

  # ------------- get lines ------------------
  lines = cv2.HoughLines(edges, rho, theta, threshold_intersect)
  lines_annotated = annotate_lines(img, lines)

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