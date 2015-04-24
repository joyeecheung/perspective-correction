#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import cv2
from operator import itemgetter
from itertools import combinations
from math import sqrt
import glob

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
  if len(lines) == 4:
    return np.array(list(set(i for j in intersections for i in j if len(i) > 0)),
                     dtype=np.float32)

  candidate_lines = [i for i in xrange(len(lines))
                     if np.sum(intersections[i:]) > 1]
  if len(candidate_lines) == 4:
    return [intersections[i][j] for i, j in
            combinations(candidate_lines, 2) if len(intersections[i][j]) > 0]

  else:
    # pick line a
    for i, line_a in enumerate(candidate_lines):
      # pick b and c that intersects a
      b_c_candidate = [j for j in candidate_lines
                       if len(intersections[j][i]) > 0 and i != j]
      for b, c in combinations(b_c_candidate, 2):
        # pick d that intersects b and c and is not a
        line_b, line_c = lines[b], lines[c]
        # store angles between a and d, b and c
        # the more they are close to 180
        # the more likely that they are the paper lines

def filter_regular(lines, margin=np.pi/18):
  print lines
  regular = np.pi/2
  count = np.zeros(len(lines))

  idxs = combinations(xrange(len(lines)), 2)

  for a, b in idxs:
    if abs(abs(lines[a][1] - lines[b][1]) - regular) < margin:
      count[a] += 1
      count[b] += 1

  return lines[count >= 2]


def eliminate_duplicates(img, lines, threshold):
  eliminated = np.zeros(lines.shape[1], dtype=bool)
  min_distance = max(img.shape) * threshold
  idxs = combinations(xrange(len(lines[0])), 2)

  for i, j in idxs:
    if eliminated[i] or eliminated[j]:
      continue
    if line_distance(lines[0][i], lines[0][j]) < min_distance:
      eliminated[j] = True

  return lines[0][eliminated == False]

def line_distance(line_a, line_b):
  rho_a, theta_a = line_a
  rho_b, theta_b = line_b
  result = np.power(rho_a, 2) + np.power(rho_b, 2)
  result -= 2*rho_b*rho_a* np.cos(theta_a - theta_b)
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


def reorder(corners, destination):
  new_corners = np.array(corners)
  for corner in corners:
    idx = np.argmin(np.sum(np.square(corner - destination), axis=1))
    new_corners[idx] = corner
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
                        threshold_distance=0.05,
                        temp=True):

# threshold_max=100
# threshold_min=10
# gaussian_blur_size=7
# median_blur_size=31
# rho=1
# theta=np.pi/180
# threshold_intersect=250
# threshold_distance=0.05
# temp = True
# ------------- get binary image -----------
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  if temp:
    gray_im = Image.fromarray(gray)

  # ------------- blur -----------------------
  # blurred = cv2.GaussianBlur(gray, (gaussian_blur_size, gaussian_blur_size), 0)
  blurred = cv2.medianBlur(gray, median_blur_size)
  if temp:
    blurred_im = Image.fromarray(blurred)

  # ------------- detect edges ---------------
  edges = cv2.Canny(blurred, threshold_min, threshold_max)
  if temp:
    edges_im = Image.fromarray(edges)
    edges_im.save('edges.jpg')

  # ------------- get lines ------------------
  lines = cv2.HoughLines(edges, rho, theta, threshold_intersect)
  lines = eliminate_duplicates(img, lines, threshold_distance)

  lines = filter_regular(lines)
  while (len(lines) < 4):
    threshold_intersect -= 10
    lines = cv2.HoughLines(edges, rho, theta, threshold_intersect)
    lines = eliminate_duplicates(img, lines, threshold_distance)
    lines = filter_regular(lines)

  cartesian = to_cartesian(img, lines)


  if temp:
    lines_annotated = Image.fromarray(annotate_lines(img, cartesian))

  intersections = get_intersections(img, cartesian)
  if temp:
    intersections_annotated = annotate_intersections(lines_annotated, intersections)
    Image.fromarray(intersections_annotated).save('intersections.jpg')

  # ------------- compute corners ------------
  corners = get_corners(lines, intersections)
  print "number of corners: ", len(corners)
  if temp:
    corners_annotated = Image.fromarray(annotate_corners(lines_annotated, corners))

  # ------------- warp ----------------------

  height, width, _ = img.shape
  min_coff = min(height, width)
  if height > width:
    new_h, new_w = int(min_coff), int(min_coff * 0.707)
  else:
    new_h, new_w = int(min_coff * 0.707), int(min_coff)
  destination = np.float32([[0, 0], [new_w, 0], [new_w, new_h], [0, new_h]])

  # sort by distance to each destination corner
  corners = reorder(corners, destination)
  trans_mat = cv2.getPerspectiveTransform(corners, destination)
  final = cv2.warpPerspective(img, trans_mat, (new_w, new_h))

  if temp:
    return (gray_im,
            blurred_im,
            edges_im,
            lines_annotated,
            corners_annotated,
            Image.fromarray(final))
  else:
    return (Image.fromarray(final),)

def check():
  images = glob.glob('../dataset/hard/*.jpg')
  # images = ['../dataset/easy/IMG_20150320_143220.jpg','../dataset/easy/IMG_20150410_091123.jpg']
  for i in images:
    print "Checking", i
    img = np.asarray(Image.open(i))
    result = correct_perspective(img)
    result[-1].save(i.replace('dataset', 'result'))

if __name__ == '__main__':
  check()