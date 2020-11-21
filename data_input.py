import cv2
import glob
import os
import numpy as np
import math
import vanishing

# Mouse callback function. Appends the x,y location of mouse click to a list.
def get_xy(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        #print(x, y)
        param.append((x, y))

# Open all image files in music directory
music_directory = "Music/"
image_file_names = glob.glob(os.path.join(music_directory, "*.png"))
for image_file_name in image_file_names:
	
	####################################################################################
	## Read Image
	####################################################################################
	
	# Open this image in grayscale
	gray_img=cv2.imread(image_file_name, 0)
	image_width = gray_img.shape[1]
	image_height = gray_img.shape[0]
	
	# Convert image to binary. Negated to make lines white for connected components
	_, bin_img = cv2.threshold(gray_img, thresh=200, maxval=255, type=cv2.THRESH_BINARY)
	bin_img = cv2.bitwise_not(bin_img)
	cv2.imshow('Music',bin_img)
	
	####################################################################################
	## Detect Staff and Bar Lines
	####################################################################################
	
	## Staff lines
	
	# Extract staff lines with morphology
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (500,1))
	staffLines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
	#cv2.imshow('Staff Lines',staffLines)
	
	# Connected components to find lines
	# 0th element is always entire image for some reason, maybe black area?
	num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(staffLines)
	
	# Get y value of each line
	yLines = np.zeros(num_labels-1)
	for i in range(num_labels-1):
		yLines[i] = stats[i+1, cv2.CC_STAT_TOP] + stats[i+1, cv2.CC_STAT_HEIGHT]/2
	#print(yLines)
	
	## Bar lines
	
	# Extract bar lines with morphology
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,100))
	barLines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
	#cv2.imshow('Bar Lines',barLines)
	
	# Connected components to find lines
	# 0th element is always entire image for some reason, maybe black area?
	num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(barLines)
	
	# Get x value of each line
	xLines = np.zeros(num_labels-1)
	for i in range(num_labels-1):
		xLines[i] = stats[i+1, cv2.CC_STAT_LEFT] + stats[i+1, cv2.CC_STAT_WIDTH]/2
	#print(xLines)
	
	## Combined
	
	# Combine staff and bar lines
	features_img = cv2.bitwise_or(staffLines,barLines)
	
	####################################################################################
	## Detect Braces
	####################################################################################
	
	# Open brace template
	template_directory = "Templates/"
	brace_filename = os.path.join(template_directory, "brace.png")
	brace_template = cv2.imread(brace_filename, 0)
	
	# Use template matching to find braces
	scores = cv2.matchTemplate(bin_img, brace_template, cv2.TM_CCOEFF_NORMED)
	thresh = 0.7
	matches = np.where(scores > thresh)
	
	# Add braces to new image
	brace_img = np.zeros([image_height, image_width], dtype=np.uint8)
	for i in range(len(matches[0])):
		x = matches[1][i]
		y = matches[0][i]
		brace_img[y:y+brace_template.shape[0], x:x+brace_template.shape[1]] |= brace_template
	
	# Add braces to detected features
	features_img = cv2.bitwise_or(features_img,brace_img)
	
	####################################################################################
	## Detect Clefs
	####################################################################################
	
	# Code
	
	####################################################################################
	## Detect Time Signatures
	####################################################################################
	
	# Code
	
	####################################################################################
	## Detect Rests
	####################################################################################
	
	# Code
	
	####################################################################################
	## Detect Notes
	####################################################################################
	
	# Show all detected features
	cv2.imshow('Detected Features', features_img)
	
	# Subtract all features above from binary image
	notes_img = bin_img - features_img
	
	# Cleanup a bit, not all features line up perfectly, and some features cut through the notes
	#size=3
	#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
	#notes_img = cv2.morphologyEx(notes_img, cv2.MORPH_CLOSE, kernel)
	#notes_img = cv2.morphologyEx(notes_img, cv2.MORPH_OPEN, kernel)
	
	cv2.imshow('Music With Features Removed', notes_img)
	
	
	
	
	
	
	
	'''kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (750,25))
	break_apart = cv2.morphologyEx(cv2.bitwise_not(gray_img), cv2.MORPH_OPEN, kernel)
	cv2.imshow('huh',break_apart)


	
	#test=gray_img.copy()
	contours,_ = cv2.findContours(staffLines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#note_locations=cv2.drawContours(test, contours, -1, (0,0,0), 1)


	cv2.imshow('remove lines',note_locations)

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	# # note_locations = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)
	# note_locations = cv2.dilate(note_locations, kernel)
	# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
	# note_locations = cv2.erode(note_locations, kernel)

	#Finds the notes through morplogy
	size=3
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
	note_locations = cv2.morphologyEx(note_locations, cv2.MORPH_CLOSE, kernel)

	note_locations = cv2.morphologyEx(note_locations, cv2.MORPH_OPEN, kernel)

	cv2.imshow('note_locations',note_locations)
	cv2.imshow('music',bin_img)'''


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
	# hougstaffLines = cv2.HougstaffLinesP(
	#     image=edge_img,
	#     rho=1,
	#     theta=math.pi/180,
	#     threshold=int(image_width * MIN_HOUGH_VOTES_FRACTION),
	#     lines=None,
	#     minLineLength=int(image_width * MIN_LINE_LENGTH_FRACTION),
	#     )
	# print("Found %d line segments" % len(hougstaffLines))

	# # For visualizing the lines, draw on a grayscale version of the image.
	# for i in range(0, len(hougstaffLines)):
	#     l = hougstaffLines[i][0]
	#     cv2.line(org_img, (l[0], l[1]), (l[2], l[3]), (0, 255, 255), 
	#              thickness=2, lineType=cv2.LINE_AA)
	# cv2.imshow("Hough linesP", org_img)
	# org_img_copy = org_img.copy()
	# vanishing.find_vanishing_point_directions(hougstaffLines, org_img_copy, num_to_find=3, K=None)
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

