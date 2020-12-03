import numpy as np
import simpleaudio as sa
import noteDictionary

noteDict, octaveNumber = noteDictionary.getDictionary()
numNotes = len(noteDict.keys())
noteDuration = 0.5

# get timesteps for each sample, T is note duration in seconds
sample_rate = 48000
songDuration = numNotes * noteDuration
song = 0

noteLabels=[
'C',
'C#',
'Db',
'D',
'D#',
'Eb',
'E' ,
'F',
'F#',
'Gb',
'G' ,
'G#',
'Ab',
'A' ,
'A#',
'Bb',
'B' ]

for i in range(len(noteLabels)):
	freq=noteDict[noteLabels[i]]
	print(freq)
	note_audio = np.linspace(0, noteDuration, int(sample_rate*noteDuration))
	note_audio = np.sin(note_audio*freq*2*np.pi)
	song = np.append(song, note_audio)

# normalize to 16-bit range
song *= 32767 / np.max(np.abs(song))/5
# convert to 16-bit data
song = song.astype(np.int16)

import scipy.io.wavfile
scipy.io.wavfile.write("song.wav", sample_rate, song)

# start playback
play_obj = sa.play_buffer(song, 1, 2, 48000)

# wait for playback to finish before exiting
play_obj.wait_done()