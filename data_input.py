import cv2
import glob
import os
import numpy as np
import simpleaudio as sa
import noteDictionary

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
		yLines[i] = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]//2
	#print(yLines)
	
	# Make sure we have a multiple of 5 staff lines
	if(len(yLines)%5!=0):
		raise('INCORRECT NUMBER OF STAFF LINES')
	
	## Bar lines
	
	# Extract bar lines with morphology
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,100))
	barLines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
	#cv2.imshow('Bar Lines',barLines)
	
	# Connected components to find lines
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
	for i in range(len(matches[0])):
		x = matches[1][i]
		y = matches[0][i]
		features_img[y:y+template.shape[0], x:x+template.shape[1]] |= template
	
	####################################################################################
	## Detect Clefs
	####################################################################################
	
	## Treble clef
	
	# Location of clefs and types. The same clef may create multiple entries in this
	yClefs = {}
	
	# Open template
	template_directory = "Templates/"
	template_filename = os.path.join(template_directory, "treble_clef.png")
	template = cv2.imread(template_filename, 0)
	
	# Use template matching to find templates
	scores = cv2.matchTemplate(bin_img, template, cv2.TM_CCOEFF_NORMED)
	thresh = 0.5
	matches = np.where(scores > thresh)
	
	# Add templates to new image
	for i in range(len(matches[0])):
		x = matches[1][i]
		y = matches[0][i]
		features_img[y:y+template.shape[0], x:x+template.shape[1]] |= template
		
		# Save location of this clef for later
		yClefs[int(y+template.shape[0]/2)] = 'treble'
	
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
	for i in range(len(matches[0])):
		x = matches[1][i]
		y = matches[0][i]
		features_img[y:y+template.shape[0], x:x+template.shape[1]] |= template
		
		# Save location of this clef for later
		yClefs[int(y+template.shape[0]/2)] = 'bass'
	
	####################################################################################
	## Label Staff Lines
	####################################################################################
	
	# Determine the note for each staff line (y values)
	yNotes = {}
	for i in range(len(yLines)//5):
		# Index of first staff line in this loop
		staffIndex = i*5
		# Number of pixels between each note
		noteSpacing = (yLines[staffIndex+4] - yLines[staffIndex]) / 8
		# Middle of these staff lines
		middle = yLines[staffIndex+2]
		
		# Determine clef of these staff lines
		thisClef = ''
		for yVal,clef in yClefs.items():
			# Skip if center of clef if not between the top and bottom staff lines
			if yVal < yLines[staffIndex] or yVal > yLines[staffIndex+4]:
				continue
			thisClef = clef
		
		# Just in case...
		if(thisClef == ''):
			raise('CLEF NOT DETECTED')
		
		if(thisClef == 'treble'):
			yNotes[int(middle-noteSpacing*9)] = ['D',6]
			yNotes[int(middle-noteSpacing*8)] = ['C',6]
			yNotes[int(middle-noteSpacing*7)] = ['B',5]
			yNotes[int(middle-noteSpacing*6)] = ['A',5]
			yNotes[int(middle-noteSpacing*5)] = ['G',5]
			yNotes[int(middle-noteSpacing*4)] = ['F',5]
			yNotes[int(middle-noteSpacing*3)] = ['E',5]
			yNotes[int(middle-noteSpacing*2)] = ['D',5]
			yNotes[int(middle-noteSpacing*1)] = ['C',5]
			yNotes[int(middle+noteSpacing*0)] = ['B',4]
			yNotes[int(middle+noteSpacing*1)] = ['A',4]
			yNotes[int(middle+noteSpacing*2)] = ['G',4]
			yNotes[int(middle+noteSpacing*3)] = ['F',4]
			yNotes[int(middle+noteSpacing*4)] = ['E',4]
			yNotes[int(middle+noteSpacing*5)] = ['D',4]
			yNotes[int(middle+noteSpacing*6)] = ['C',4]
			yNotes[int(middle+noteSpacing*7)] = ['B',3]
			yNotes[int(middle+noteSpacing*8)] = ['A',3]
			yNotes[int(middle+noteSpacing*8)] = ['G',3]
		
		elif(thisClef == 'bass'):
			yNotes[int(middle-noteSpacing*9)] = ['F',4]
			yNotes[int(middle-noteSpacing*8)] = ['E',4]
			yNotes[int(middle-noteSpacing*7)] = ['D',4]
			yNotes[int(middle-noteSpacing*6)] = ['C',4]
			yNotes[int(middle-noteSpacing*5)] = ['B',3]
			yNotes[int(middle-noteSpacing*4)] = ['A',3]
			yNotes[int(middle-noteSpacing*3)] = ['G',3]
			yNotes[int(middle-noteSpacing*2)] = ['F',3]
			yNotes[int(middle-noteSpacing*1)] = ['E',3]
			yNotes[int(middle+noteSpacing*0)] = ['D',3]
			yNotes[int(middle+noteSpacing*1)] = ['C',3]
			yNotes[int(middle+noteSpacing*2)] = ['B',2]
			yNotes[int(middle+noteSpacing*3)] = ['A',2]
			yNotes[int(middle+noteSpacing*4)] = ['G',2]
			yNotes[int(middle+noteSpacing*5)] = ['F',2]
			yNotes[int(middle+noteSpacing*6)] = ['E',2]
			yNotes[int(middle+noteSpacing*7)] = ['D',2]
			yNotes[int(middle+noteSpacing*8)] = ['C',2]
			yNotes[int(middle+noteSpacing*8)] = ['B',1]
	
	####################################################################################
	## Detect Time Signatures
	####################################################################################
	
	# Assume 44 time unless we find a template match that says otherwise
	timeSignature = '44'
	
	## 44 time
	
	# Open template
	template_directory = "Templates/"
	template_filename = os.path.join(template_directory, "time_44.png")
	template = cv2.imread(template_filename, 0)
	
	# Use template matching to find templates
	scores = cv2.matchTemplate(bin_img, template, cv2.TM_CCOEFF_NORMED)
	thresh = 0.9
	matches = np.where(scores > thresh)
	
	# Add templates to new image
	for i in range(len(matches[0])):
		x = matches[1][i]
		y = matches[0][i]
		features_img[y:y+template.shape[0], x:x+template.shape[1]] |= template
		timeSignature = '44'
	
	## 38 time
	
	# Open template
	template_directory = "Templates/"
	template_filename = os.path.join(template_directory, "time_38.png")
	template = cv2.imread(template_filename, 0)
	
	# Use template matching to find templates
	scores = cv2.matchTemplate(bin_img, template, cv2.TM_CCOEFF_NORMED)
	thresh = 0.9
	matches = np.where(scores > thresh)
	
	# Add templates to new image
	for i in range(len(matches[0])):
		x = matches[1][i]
		y = matches[0][i]
		features_img[y:y+template.shape[0], x:x+template.shape[1]] |= template
		timeSignature = '38'
	
	####################################################################################
	## Detect Rests
	####################################################################################
	
	# Code
	
	####################################################################################
	## Detect Short Staff Lines
	####################################################################################
	
	# Open template
	template_directory = "Templates/"
	template_filename = os.path.join(template_directory, "short_staff.png")
	template = cv2.imread(template_filename, 0)
	
	# Use template matching to find templates
	scores = cv2.matchTemplate(bin_img, template, cv2.TM_CCOEFF_NORMED)
	thresh = 0.9
	matches = np.where(scores > thresh)
	
	# Add templates to new image
	for i in range(len(matches[0])):
		x = matches[1][i]
		y = matches[0][i]
		features_img[y:y+template.shape[0], x:x+template.shape[1]] |= template
	
	####################################################################################
	## Detect Title
	####################################################################################
	
	# Basically just does away with all the pixels above first staff line
	# Determine threshold above highest staff line
	threshY = int(yLines[0] - (yLines[5]-yLines[0]))
	
	# Add title to new image
	title_img = np.zeros([image_height, image_width], dtype=np.uint8)
	title_img[0:threshY,:] = bin_img[0:threshY,:]
	
	# Add title to detected features
	features_img = cv2.bitwise_or(features_img,title_img)
	
	####################################################################################
	## Detect Notes
	####################################################################################
	
	## Cleanup
	
	# Subtract all features above from binary image
	bin_img = cv2.bitwise_and(bin_img, cv2.bitwise_not(features_img))
	
	# Cleanup a bit, not all features line up perfectly, and some features cut through the notes
	kernel = np.ones([4, 1])
	bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
	
	cv2.imshow('Music With Features Removed', bin_img)
	
	## Detect whole notes
	
	# Location of note and number of beats. The same note may create multiple entries in this
	noteBeats = []
	
	# Open template
	template_directory = "Templates/"
	template_filename = os.path.join(template_directory, "note_whole.png")
	template = cv2.imread(template_filename, 0)
	
	# Use template matching to find templates
	scores = cv2.matchTemplate(bin_img, template, cv2.TM_CCOEFF_NORMED)
	thresh = 0.7
	matches = np.where(scores > thresh)
	
	# Add templates to new image
	notes_img = np.zeros([image_height, image_width], dtype=np.uint8)
	for i in range(len(matches[0])):
		x = matches[1][i]
		y = matches[0][i]
		notes_img[y:y+template.shape[0], x:x+template.shape[1]] |= template
		
		# Save duration of this note for later
		noteBeats.append([int(x+template.shape[1]/2), int(y+template.shape[0]/2), 4])
	
	# Remove these notes from image
	bin_img = cv2.bitwise_and(bin_img, cv2.bitwise_not(notes_img))
	
	## Detect half notes
	
	# Open template
	template_directory = "Templates/"
	template_filename = os.path.join(template_directory, "note_half.png")
	template = cv2.imread(template_filename, 0)
	
	# Use template matching to find templates
	scores = cv2.matchTemplate(bin_img, template, cv2.TM_CCOEFF_NORMED)
	thresh = 0.5
	matches = np.where(scores > thresh)
	
	# Add templates to new image
	for i in range(len(matches[0])):
		x = matches[1][i]
		y = matches[0][i]
		notes_img[y:y+template.shape[0], x:x+template.shape[1]] |= template
		
		# Save duration of this note for later
		noteBeats.append([int(x+template.shape[1]/2), int(y+template.shape[0]/2), 2])
	
	# Remove these notes from image
	bin_img = cv2.bitwise_and(bin_img, cv2.bitwise_not(notes_img))
	
	## Detect quarter notes
	
	# Open template
	template_directory = "Templates/"
	template_filename = os.path.join(template_directory, "note_quarter.png")
	template = cv2.imread(template_filename, 0)
	
	# Use template matching to find templates
	scores = cv2.matchTemplate(bin_img, template, cv2.TM_CCOEFF_NORMED)
	thresh = 0.7
	matches = np.where(scores > thresh)
	
	# Add templates to new image
	for i in range(len(matches[0])):
		x = matches[1][i]
		y = matches[0][i]
		notes_img[y:y+template.shape[0], x:x+template.shape[1]] |= template
		
		# Save duration of this note for later
		noteBeats.append([int(x+template.shape[1]/2), int(y+template.shape[0]/2), 1])
	
	# Remove these notes from image
	bin_img = cv2.bitwise_and(bin_img, cv2.bitwise_not(notes_img))
	
	# Show all detected features
	features_img = cv2.bitwise_or(features_img,notes_img)
	cv2.imshow('Detected Features', features_img)
	cv2.imshow('Detected Notes', notes_img)
	
	## Determine note durations and frequencies
	
	noteBeats = np.array(noteBeats)
	
	# Connected components to find lines
	num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(notes_img)
	
	# 0th element is background label, need to ignore it
	num_labels -= 1
	stats = np.delete(stats, 0, 0)
	
	
	
	## Determine location of each note
	
	note_locations = np.zeros([num_labels, 2])
	
	# Find location of all notes
	for i in range(num_labels):
		note_locations[i,0] = int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]/2)
		note_locations[i,1] = int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]/2)
	
	# Separate notes into groups based on which staff lines they belong to
	numStaffGroups = len(yLines)//5
	noteLocationGroups = np.zeros(numStaffGroups, dtype=object)
	for i in range(numStaffGroups):
		if(i < numStaffGroups-1): # All but last group
			# Define threshold y value between staff line groups
			staffIndex = i*5
			thresh = (yLines[staffIndex+4]+yLines[staffIndex+5])/2
			
			# Get index of each note above threshold
			indices = np.where(note_locations[:,1]<thresh)[0]
			
			# Add notes to group
			noteLocationGroups[i] = note_locations[indices,:]
			
			# Remove notes that have been grouped
			note_locations = np.delete(note_locations, indices, 0)
		else: # Last group
			# Add notes to group
			noteLocationGroups[i] = note_locations
		
		# Sort notes by x value to order sequentially
		noteLocationGroups[i] = noteLocationGroups[i][np.argsort(noteLocationGroups[i][:,0])]
	
	
	
	
	
	
	
	
	####################################################################################
	## Create Song
	####################################################################################
	
	# Song info
	numBeats = 32
	bpm = 180
	beatDuration = 60/bpm # Seconds
	samplingRate = 48000 # Hz
	beatSamples = int(beatDuration * samplingRate)
	songDuration = beatDuration * numBeats # Seconds
	songSamples = int(songDuration*samplingRate)
	song = np.zeros(songSamples)
	
	# Dictionary of notes
	dictOctave = 4
	noteDictionary = {
		'C'	:261.63,
		'D' :293.66,
		'E' :329.63,
		'F' :349.23,
		'G' :392.00,
		'A' :440.00,
		'B' :493.88}
	
	# Easier format for finding y value of notes
	yStaffLines = np.array(list(yNotes.keys()))
	#print(noteBeats)
	# Loop through all note groups
	for i in range(len(noteLocationGroups)):
		# Time at which this note group begins
		if(i//2 == 0):
			noteSampleStart = 0
		else:
			noteSampleStart = 16*beatSamples
		
		# Loop through all notes in this group
		for j in range(len(noteLocationGroups[i])):
			# Get number of beats for this note
			x,y = noteLocationGroups[i][j,:]
			closestNote = np.copy(noteBeats[:,0:2])
			closestNote[:,0] -= int(x)
			closestNote[:,1] -= int(y)
			closestNote = np.linalg.norm(closestNote, axis=1)
			durationIndex = np.argmin(closestNote)
			numBeats = noteBeats[durationIndex,2]
			
			# Find index of staff line that is closest to this note
			noteIndex = np.argmin(np.abs(yStaffLines - noteLocationGroups[i][j,1]))
			
			# Determine frequency of this note
			noteLetter, octave = yNotes[yStaffLines[noteIndex]]
			note_freq = noteDictionary[noteLetter]*2**(octave-dictOctave)
			
			# If sharp
			#note_freq = note_freq*2**(1/12)
			# If flat
			#note_freq = note_freq*2**(-1/12)
			
			# Create sinusoid for this note
			t = np.linspace(0, numBeats*beatDuration, numBeats*beatSamples)
			noteSampleEnd = noteSampleStart + numBeats*beatSamples
			decay = -3
			song[noteSampleStart : noteSampleEnd] += np.sin(t*note_freq*2*np.pi) * np.exp(t*decay)
			noteSampleStart = noteSampleEnd
	
	## Play music
	
	# normalize to 16-bit range
	song *= 32767 / np.max(np.abs(song))
	song /= 5 # Not too loud
	# convert to 16-bit data
	song = song.astype(np.int16)
	
	import scipy.io.wavfile
	scipy.io.wavfile.write("song.wav", samplingRate, song)
	
	while(True):
		if(cv2.waitKey(0) != ord('p')):
			break
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


