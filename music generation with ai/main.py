import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from music21 import stream, note, midi
import mido

# Assume 'sequences' is your preprocessed data (list of lists, each inner list is a sequence of note integers)
# Example: sequences = [[60, 62, 64], [62, 64, 65], ...]  # MIDI note numbers

# Hyperparameters
seq_length = 50  # Length of input sequences
vocab_size = 128  # MIDI notes (0-127)

# Prepare data
X = []
y = []
for seq in sequences:
    for i in range(len(seq) - seq_length):
        X.append(seq[i:i+seq_length])
        y.append(seq[i+seq_length])
X = np.array(X)
y = np.array(y)

# Build model
model = Sequential([
    LSTM(256, input_shape=(seq_length, 1), return_sequences=True),
    Dropout(0.2),
    LSTM(256),
    Dense(vocab_size, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Train (this may take time; use GPU if available)
model.fit(X, y, epochs=20, batch_size=64)

# Generate new melody
seed = np.random.randint(0, vocab_size, seq_length).reshape(1, seq_length, 1)
generated = []
for _ in range(100):  # Generate 100 notes
    pred = model.predict(seed, verbose=0)
    next_note = np.argmax(pred)
    generated.append(next_note)
    seed = np.append(seed[:, 1:, :], [[[next_note]]], axis=1)

# Convert to MIDI
s = stream.Stream()
for n in generated:
    s.append(note.Note(n))
mf = midi.translate.streamToMidiFile(s)
mf.open('generated_music.mid', 'wb')
mf.write()
mf.close()

# Playback (optional, requires mido and a MIDI player)
mid = mido.MidiFile('generated_music.mid')
for msg in mid.play():
    pass  # Plays the file
