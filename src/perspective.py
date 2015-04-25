#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import cv2
from itertools import combinations

def get_intersections(img, lines):
  height, width, _ = img.shape
  line_count = len(lines)
  intersections = [[[] for i in xrange(line_count)] for j in xrange(line_count)]

  for i, pointsa in enumerate(lines):
    x1, y1, x2, y2 = pointsa

    for j, pointsb in enumerate(lines):
      if len(intersections[i][j]) > 0:
        continue
      x3, y3, x4, y4 = pointsb

      d = (x1-x2) * (y3-y4) - (y1-y2) * (x3-x4)

      if d != 0:
        x = ((x1*y2 - y1*x2) * (x3-x4) - (x1-x2) * (x3*y4 - y3*x4)) / d
        y = ((x1*y2 - y1*x2) * (y3-y4) - (y1-y2) * (x3*y4 - y3*x4)) / d

        if 0 <= x and x <= width and 0 <= y and y <= height:
          intersections[i][j] = (x, y)
          intersections[j][i] = (x, y)

  return intersections

def annotate_intersections(img, intersections):
  temp = np.array(img)
  for row in intersections:
    for col in row:
      if len(col) > 0:
        cv2.circle(temp, col, 20, (255, 0, 0), 10)

  return temp

def annotate_corners(img, corners):
  temp = np.array(img)
  for x, y in corners:
    cv2.circle(temp, (x, y), 20, (0, 255, 0), 10)
  return temp

def get_corners(lines, intersections):
  return np.array(list(set(i for j in intersections for i in j if len(i) > 0)),
                  dtype=np.float32)[:4]

def filter_regular(lines, margin=np.pi/18):
  regular = np.pi/2
  count = np.zeros(len(lines))

  idxs = combinations(xrange(len(lines)), 2)

  for a, b in idxs:
    if abs(abs(lines[a][1] - lines[b][1]) - regular) < margin:
      count[a] += 1
      count[b] += 1

  return lines[count >= 2]

def eliminate_duplicates(img, lines, threshold):
  eliminated = np.zeros(len(lines), dtype=bool)
  min_distance = max(img.shape) * threshold
  min_theta = np.pi * threshold
  idxs = combinations(xrange(len(lines)), 2)

  for i, j in idxs:
    if eliminated[i] or eliminated[j]:
      continue

    line_a, line_b = lines[i], lines[j]
    theta_diff = abs(line_a[1] - line_b[1])
    if theta_diff > np.pi / 2:
      theta_diff = np.pi - theta_diff
    
    if line_distance(line_a, line_b) < min_distance and theta_diff < min_theta:
      eliminated[i] = True

  return lines[eliminated == False]

def line_distance(line_a, line_b):
  rho_a, theta_a = line_a
  rho_b, theta_b = line_b

  result = rho_a ** 2 + rho_b ** 2
  result -= 2*rho_b*rho_a * np.cos(theta_a - theta_b)
  return np.sqrt(result)

def to_cartesian(img, lines):
  height, width, _ = img.shape
  coff = max(height, width)
  cartesian = []
  for rho, theta in lines:
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    x1, y1 = int(x0 + coff * (-b)), int(y0 + coff * (a))
    x2, y2 = int(x0 - coff * (-b)), int(y0 - coff * (a))
    cartesian.append((x1, y1, x2, y2))
  return cartesian

def annotate_lines(img, lines):
  annnotated = np.array(img)

  for x1, y1, x2, y2 in lines:
    cv2.line(annnotated, (x1, y1), (x2, y2), (0, 0, 255), 10)

  return annnotated.astype(np.uint8)

def reorder(corners):
  new_corners = np.array(corners)
  mean = np.mean(corners, axis=0)

  for corner in corners:
    if corner[0] < mean[0] and corner[1] < mean[1]: # upper-left
      new_corners[0] = corner
    if corner[0] > mean[0] and corner[1] < mean[1]: # upper-right
      new_corners[1] = corner
    if corner[0] > mean[0] and corner[1] > mean[1]: # lower-right
      new_corners[2] = corner
    if corner[0] < mean[0] and corner[1] > mean[1]: # lower-left
      new_corners[3] = corner

  return new_corners

def show(img):
  Image.fromarray(img).show()

def correct_perspective(img, threshold_max=140,
                        threshold_min=30,
                        gaussian_blur_size=7,
                        median_blur_size=51,
                        rho=1,
                        theta=np.pi/180,
                        threshold_intersect=250,
                        threshold_distance=0.15,
                        intermediate=True):

  # ------------- get binary image -----------
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  if intermediate:
    gray_im = Image.fromarray(gray)

  # ------------- blur -----------------------
  # blurred = cv2.GaussianBlur(gray, (gaussian_blur_size, gaussian_blur_size), 0)
  blurred = cv2.medianBlur(gray, median_blur_size)
  if intermediate:
    blurred_im = Image.fromarray(blurred)

  # ------------- detect edges ---------------
  edges = cv2.Canny(blurred, threshold_min, threshold_max)
  if intermediate:
    edges_im = Image.fromarray(edges)

  # ------------- get lines ------------------
  lines = cv2.HoughLines(edges, rho, theta, threshold_intersect)

  lines = filter_regular(lines[0])
  lines = eliminate_duplicates(img, lines, threshold_distance)

  while (len(lines) < 4):
    threshold_intersect -= 10
    lines = cv2.HoughLines(edges, rho, theta, threshold_intersect)
    lines = filter_regular(lines[0])
    lines = eliminate_duplicates(img, lines, threshold_distance)

  cartesian = to_cartesian(img, lines)

  if intermediate:
    lines_annotated = Image.fromarray(annotate_lines(img, cartesian))

  intersections = get_intersections(img, cartesian)

  # ------------- compute corners ------------
  corners = get_corners(lines, intersections)
  print "number of corners: ", len(corners)
  if intermediate:
    corners_annotated = Image.fromarray(annotate_corners(lines_annotated, corners))

  # ------------- warp ----------------------
  height, width, _ = img.shape
  min_coff = min(height, width)
  if height > width:
    new_h, new_w = int(min_coff), int(min_coff * 0.707)
  else:
    new_h, new_w = int(min_coff * 0.707), int(min_coff)
  destination = np.float32([[0, 0], [new_w, 0], [new_w, new_h], [0, new_h]])

  corners = reorder(corners)
  trans_mat = cv2.getPerspectiveTransform(corners, destination)
  final = cv2.warpPerspective(img, trans_mat, (new_w, new_h))

  if intermediate:
    return (gray_im,
            blurred_im,
            edges_im,
            lines_annotated,
            corners_annotated,
            Image.fromarray(final))
  else:
    return (Image.fromarray(final),)
