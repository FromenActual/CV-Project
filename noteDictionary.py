def getDictionary():
	# Create dictionary of all notes in 4th octave
	noteDictionary = {
		'C'	:261.63,
		'C#':277.18,
		'Db':277.18,
		'D' :293.66,
		'D#':311.13,
		'Eb':311.135,
		'E' :329.63,
		'F' :349.23,
		'F#':369.99,
		'Gb':369.99,
		'G' :392.00,
		'G#':415.30,
		'Ab':415.30,
		'A' :440.00,
		'A#':466.16,
		'Bb':466.16,
		'B' :493.88}
	
	# Return dictionary and octave number
	return [noteDictionary, 4]