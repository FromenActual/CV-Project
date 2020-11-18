import cv2
import glob
import os
import numpy as np
import math
import vanishing
############################################################################
#Currently reads in image with org_img with 3 color dim
#and gray_img as a grey img
QUERY_IMAGE_DIRECTORY="/home/gordon/SchoolAssingments/EENG507/project/data"
image_file_names = glob.glob(os.path.join(QUERY_IMAGE_DIRECTORY, "*.png"))
org_img=cv2.imread(image_file_names[0])
gray_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
image_width = org_img.shape[1]
image_height = org_img.shape[0]
gray_img = cv2.bitwise_not(gray_img)
test=gray_img.copy()
############################################################################
#Finds the notes through morplogy
size=3
# Remove horizontal lines
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (500,1))
lines = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
cv2.imshow('what',lines)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (750,25))
break_apart = cv2.morphologyEx(cv2.bitwise_not(gray_img), cv2.MORPH_OPEN, kernel)
cv2.imshow('huh',break_apart)


contours,_ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
note_locations=cv2.drawContours(test, contours, -1, (0,0,0), 1)


cv2.imshow('remove lines',note_locations)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# # note_locations = cv2.morphologyEx(org_img, cv2.MORPH_CLOSE, kernel)
# note_locations = cv2.dilate(note_locations, kernel)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
# note_locations = cv2.erode(note_locations, kernel)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
note_locations = cv2.morphologyEx(note_locations, cv2.MORPH_CLOSE, kernel)

note_locations = cv2.morphologyEx(note_locations, cv2.MORPH_OPEN, kernel)

cv2.imshow('note_locations',note_locations)
cv2.imshow('music',org_img)


# ############################################################################
# #Does the hough transform stuff. does not work well
# thresh_canny=5
# MIN_FRACT_EDGES = 0.05
# MAX_FRACT_EDGES = 0.1
# edge_img = cv2.Canny(
# 	image=gray_img,
# 	apertureSize=3,  # size of Sobel operator
# 	threshold1=thresh_canny,  # lower threshold
# 	threshold2=3 * thresh_canny,  # upper threshold
# 	L2gradient=True)  # use more accurate L2 norm
# while np.sum(edge_img)/255 < MIN_FRACT_EDGES * (image_width * image_height):
# 	thresh_canny *= 0.9
# 	edge_img = cv2.Canny(
# 		image=gray_img,apertureSize=3,  # size of Sobel operator
# 		threshold1=thresh_canny,  # lower threshold
# 		threshold2=3 * thresh_canny,  # upper threshold
# 		L2gradient=True)  # use more accurate L2 norm
# while np.sum(edge_img)/255 > MAX_FRACT_EDGES * (image_width * image_height):
# 	thresh_canny *= 1.1
# 	edge_img = cv2.Canny(image=gray_img,apertureSize=3,  # size of Sobel operator
# 	threshold1=thresh_canny,  # lower threshold
# 	threshold2=3 * thresh_canny,  # upper threshold
# 	L2gradient=True)  # use more accurate L2 norm
# cv2.imshow('Edge Image',edge_img)
# MIN_HOUGH_VOTES_FRACTION=.05
# MIN_LINE_LENGTH_FRACTION=.05
# houghLines = cv2.HoughLinesP(
#     image=edge_img,
#     rho=1,
#     theta=math.pi/180,
#     threshold=int(image_width * MIN_HOUGH_VOTES_FRACTION),
#     lines=None,
#     minLineLength=int(image_width * MIN_LINE_LENGTH_FRACTION),
#     )
# print("Found %d line segments" % len(houghLines))

# # For visualizing the lines, draw on a grayscale version of the image.
# for i in range(0, len(houghLines)):
#     l = houghLines[i][0]
#     cv2.line(org_img, (l[0], l[1]), (l[2], l[3]), (0, 255, 255), 
#              thickness=2, lineType=cv2.LINE_AA)
# cv2.imshow("Hough linesP", org_img)
# org_img_copy = org_img.copy()
# vanishing.find_vanishing_point_directions(houghLines, org_img_copy, num_to_find=3, K=None)
# cv2.imshow("overlay", org_img_copy)

# ############################################################################
# # https://docs.opencv.org/3.4/dd/dd7/tutorial_morph_lines_detection.html


# horizontal = np.copy(edge_img)
# vertical = np.copy(edge_img)
# # [init]
# # [horiz]
# # Specify size on horizontal axis
# cols = horizontal.shape[1]
# horizontal_size = cols // 30
# # Create structure element for extracting horizontal lines through morphology operations
# horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
# # Apply morphology operations
# horizontal = cv2.erode(horizontal, horizontalStructure)
# horizontal = cv2.dilate(horizontal, horizontalStructure)
# # Show extracted horizontal lines
# rows = vertical.shape[0]
# verticalsize = rows // 30
# # Create structure element for extracting vertical lines through morphology operations
# verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
# # Apply morphology operations
# vertical = cv2.erode(vertical, verticalStructure)
# vertical = cv2.dilate(vertical, verticalStructure)
# # Show extracted vertical lines
# cv2.imshow("vertical", vertical)
# # [vert]
# # [smooth]
# # Inverse vertical image
# vertical = cv2.bitwise_not(vertical)
# cv2.imshow("vertical_bit", horizontal)
# '''
# Extract edges and smooth image according to the logic
# 1. extract edges
# 2. dilate(edges)
# 3. src.copyTo(smooth)
# 4. blur smooth img
# 5. smooth.copyTo(src, edges)
# '''
# # Step 1
# edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
#                             cv2.THRESH_BINARY, 3, -2)
# cv2.imshow("edges", edges)
# # Step 2
# kernel = np.ones((2, 2), np.uint8)
# edges = cv2.dilate(edges, kernel)
# cv2.imshow("dilate", edges)
# # Step 3
# smooth = np.copy(vertical)
# # Step 4
# smooth = cv2.blur(smooth, (2, 2))
# # Step 5
# (rows, cols) = np.where(edges != 0)
# vertical[rows, cols] = smooth[rows, cols]
# cv2.imshow("horizontal", vertical)
# ############################################################################
# #This shows promise
# size=2
# kernel = kernel = np.ones((size, size), np.uint8)
# horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, kernel)
# horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, kernel)
# cv2.imshow('test',horizontal)

cv2.waitKey(0)

