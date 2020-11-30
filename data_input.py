import cv2
import glob
import os
import numpy as np
import simpleaudio as sa
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
	num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(staffLines)
	
	# 0th element is background label, need to ignore it
	num_labels -= 1
	stats = np.delete(stats, 0, 0)
	
	# Get y value of each line
	yLines = np.zeros(num_labels)
	for i in range(num_labels):
		yLines[i] = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]/2
	#print(yLines)
	
	## Bar lines
	
	# Extract bar lines with morphology
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,100))
	barLines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
	#cv2.imshow('Bar Lines',barLines)
	
	# Connected components to find lines
	# 0th element is background label, need to ignore it
	num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(barLines)
	
	# 0th element is background label, need to ignore it
	num_labels -= 1
	stats = np.delete(stats, 0, 0)
	
	# Get x value of each line
	xLines = np.zeros(num_labels)
	for i in range(num_labels):
		xLines[i] = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]/2
	#print(xLines)
	
	## Combined
	
	# Combine staff and bar lines
	features_img = cv2.bitwise_or(staffLines,barLines)
	
	# Subtract all features above from binary image
	bin_img = cv2.bitwise_and(bin_img, cv2.bitwise_not(features_img))
	
	####################################################################################
	## Detect Braces
	####################################################################################
	
	# Open template
	template_directory = "Templates/"
	template_filename = os.path.join(template_directory, "brace.png")
	template = cv2.imread(template_filename, 0)
	
	# Use template matching to find templates
	scores = cv2.matchTemplate(bin_img, template, cv2.TM_CCOEFF_NORMED)
	thresh = 0.5
	matches = np.where(scores > thresh)
	
	# Add templates to new image
	template_img = np.zeros([image_height, image_width], dtype=np.uint8)
	for i in range(len(matches[0])):
		x = matches[1][i]
		y = matches[0][i]
		template_img[y:y+template.shape[0], x:x+template.shape[1]] |= template
	
	# Add templates to detected features
	features_img = cv2.bitwise_or(features_img,template_img)
	
	####################################################################################
	## Detect Clefs
	####################################################################################
	
	## Treble clef
	
	# Open template
	template_directory = "Templates/"
	template_filename = os.path.join(template_directory, "treble_clef.png")
	template = cv2.imread(template_filename, 0)
	
	# Use template matching to find templates
	scores = cv2.matchTemplate(bin_img, template, cv2.TM_CCOEFF_NORMED)
	thresh = 0.5
	matches = np.where(scores > thresh)
	
	# Add templates to new image
	template_img = np.zeros([image_height, image_width], dtype=np.uint8)
	for i in range(len(matches[0])):
		x = matches[1][i]
		y = matches[0][i]
		template_img[y:y+template.shape[0], x:x+template.shape[1]] |= template
	
	# Add templates to detected features
	features_img = cv2.bitwise_or(features_img,template_img)
	
	## Bass clef
	
	# Open template
	template_directory = "Templates/"
	template_filename = os.path.join(template_directory, "bass_clef.png")
	template = cv2.imread(template_filename, 0)
	
	# Use template matching to find templates
	scores = cv2.matchTemplate(bin_img, template, cv2.TM_CCOEFF_NORMED)
	thresh = 0.5
	matches = np.where(scores > thresh)
	
	# Add templates to new image
	template_img = np.zeros([image_height, image_width], dtype=np.uint8)
	for i in range(len(matches[0])):
		x = matches[1][i]
		y = matches[0][i]
		template_img[y:y+template.shape[0], x:x+template.shape[1]] |= template
	
	# Add templates to detected features
	features_img = cv2.bitwise_or(features_img,template_img)
	
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
	
	# Subtract all features above from binary image
	bin_img = cv2.bitwise_and(bin_img, cv2.bitwise_not(features_img))
	
	# Cleanup a bit, not all features line up perfectly, and some features cut through the notes
	'''size=2
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
	bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
	# Reconnect notes that got disconnected
	size=4
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
	bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)'''
	
	cv2.imshow('Music With Features Removed', bin_img)
	
	# Open template
	template_directory = "Templates/"
	template_filename = os.path.join(template_directory, "note.png")
	template = cv2.imread(template_filename, 0)
	
	# Use template matching to find templates
	scores = cv2.matchTemplate(bin_img, template, cv2.TM_CCOEFF_NORMED)
	thresh = 0.5
	matches = np.where(scores > thresh)
	
	# Add templates to new image
	template_img = np.zeros([image_height, image_width], dtype=np.uint8)
	for i in range(len(matches[0])):
		x = matches[1][i]
		y = matches[0][i]
		template_img[y:y+template.shape[0], x:x+template.shape[1]] |= template
		
	cv2.imshow('Detected Notes', template_img)
	
	# Add templates to detected features
	features_img = cv2.bitwise_or(features_img,template_img)
	
	# Show all detected features
	cv2.imshow('Detected Features', features_img)
	
	## Detect notes
	
	# Connected components to find lines
	# 0th element is background label, need to ignore it
	num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(template_img)
	
	# 0th element is background label, need to ignore it
	num_labels -= 1
	stats = np.delete(stats, 0, 0)
	
	# Song info
	num_labels = 11 # First row for now
	numNotes = num_labels
	samplingRate = 48000 # Hz
	noteDuration = 0.5 # Seconds
	noteSamples = int(noteDuration * samplingRate)
	songDuration = noteDuration * numNotes # Seconds
	songSamples = int(songDuration*samplingRate)
	song = np.linspace(0, songDuration, songSamples)
	
	# (x,y) location of each note
	note_locations = np.zeros([num_labels, 2])
	
	# Find location of all notes
	for i in range(num_labels):
		note_locations[i,0] = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]/2
		note_locations[i,1] = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]/2
	
	# Sort notes by x value to order sequentially
	note_locations = note_locations[np.argsort(note_locations[:,0])]
	
	staffSpacing = (yLines[4]-yLines[0])/4
	noteSpacing = staffSpacing/2
	
	A_freq = 440
	A_yVal = np.mean(yLines[2:4])
	print(A_yVal)
	
	# Loop through notes and
	for i in range(num_labels):
		foo = round((note_locations[i,1] - A_yVal)/noteSpacing)
		
		print(foo)
		
		
		note_audio = np.linspace(0, noteDuration, noteSamples)
		note_freq = A_freq * (1-foo/12)
		song[i*(noteSamples) : (i+1)*noteSamples] = np.sin(note_audio*note_freq*2*np.pi)
	#print(note_locations)
	
	
	
	# Show images
	cv2.waitKey(1)
	
	# normalize to 16-bit range
	song *= 32767 / np.max(np.abs(song))
	# convert to 16-bit data
	song = song.astype(np.int16)
	
	import scipy.io.wavfile
	scipy.io.wavfile.write("song.wav", samplingRate, song)

	# start playback
	play_obj = sa.play_buffer(song, 1, 2, samplingRate)

	# wait for playback to finish before exiting
	play_obj.wait_done()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	


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
	#     theta=np.pi/180,
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

