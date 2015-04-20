#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import cv2

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

def annotate_intersections(lines_annotated, intersections):
  temp = np.array(lines_annotated)
  for row in intersections:
    for col in row:
      if len(col) > 0:
        cv2.circle(temp, col, 20, (255, 0, 0), 10)

  return temp

def get_corners(lines, intersections):
  candidate_lines = [i for i in xrange(len(lines))
                     if np.sum(intersections[i:] > 1]
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

def eliminate_duplicates(img, lines, threshold):
  eliminated = [False for i in xrange(len(lines[0]))]
  min_distance = max(img.shape) * threshold

  for i, line_a in enumerate(lines[0]):
    if eliminated[i]:
      continue

    for j, line_b in enumerate(lines[0]):
      if eliminated[j]:
        continue

      if distance(line_a, line_b) < min_distance and i != j:
        eliminated[j] = True

  return [lines[0][i] for i in xrange(len(lines[0])) if not eliminated[i]]

def distance(line_a, line_b):
  rho_a, theta_a = line_a
  rho_b, theta_b = line_b
  result = np.pow(rho_a, 2) + np.pow(rho_b, 2)
  result -= 2*rho_b*rho_a* np.cos(theta_a - theta_b)
  return result

def to_cartesian(img, lines):
  height, width, _ = img.shape
  coff = max(height, width)
  cartesian = []
  for rho, theta in lines:
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    print "x0 = %f, y0 = %f, rho = %f, theta = %f" %(x0, y0, rho, theta)
    x1, y1 = int(x0 + coff * (-b)), int(y0 + coff * (a))
    x2, y2 = int(x0 - coff * (-b)), int(y0 - coff * (a))
    cartesian.append((x1, y1, x2, y2))
  return cartesian

def annotate_lines(img, lines):
  annnotated = np.array(img)

  for x1, y1, x2, y2 in lines:
    cv2.line(annnotated, (x1, y1), (x2, y2), (0, 0, 255), 10)

  return annnotated.astype(np.uint8)

def eliminate_duplicates(lines):

def correct_perspective(img, threshold_max=200,
                        threshold_min=60,
                        gaussian_blur_size=7,
                        median_blur_size=31
                        rho=1,
                        theta=np.pi/180,
                        threshold_intersect=400,
                        threshold_distance=0.5):
  # ------------- get binary image -----------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_im = Image.fromarray(gray)

# ------------- blur -----------------------
# blurred = cv2.GaussianBlur(gray, (gaussian_blur_size, gaussian_blur_size), 0)
blurred = cv2.medianBlur(gray, median_blur_size)
blurred_im = Image.fromarray(blurred)

# ------------- detect edges ---------------
edges = cv2.Canny(blurred, threshold_min, threshold_max)
edges_im = Image.fromarray(edges)

# ------------- get lines ------------------
lines = cv2.HoughLines(edges, rho, theta, threshold_intersect)
lines = eliminate_duplicates(img, lines, threshold_distance)
cartesian = to_cartesian(img, lines)
lines_annotated = annotate_lines(img, cartesian)

intersections = get_intersections(img, cartesian)
intersections_annotated = annotate_intersections(lines_annotated, intersections)

Image.fromarray(intersections_annotated).show()
  # ------------- compute corners ------------
  corners = ...
  corners_annotated = ...

  # ------------- warp ----------------------

  # get trasnformation matrix

  # warp the image
  final = ...

  return (gray_im,
          blurred_im,
          edges_im,
          lines_annotated,
          corners_annotated,
          Image.fromarray(final))