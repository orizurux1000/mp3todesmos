import sys
import json
import librosa
import numpy as np


def main():
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} mp3_path.mp3 output_path.txt')
        sys.exit(1)
    
    data = convert(sys.argv[1])[:50]
    output = '['
    polygon_strings = []
    for polygon in data:
        points = ','.join(f'({tone[0]},{tone[1]})' for tone in polygon)
        polygon_strings.append(f'polygon({points})')
    output += ','.join(polygon_strings)
    output += ']'

    with open(sys.argv[2], "w") as file:
        file.write(output)


def convert(path, n_fft=256):
    y, sr = librosa.load(path, sr=None)
    D = librosa.stft(y, n_fft=n_fft, hop_length=128)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return [
        [[float(f), float(m)] for f, m in zip(freqs, np.abs(D[:, i])) if m >= 0.0001 and f <= 8000]
        for i in range(D.shape[1])
    ]

    
if __name__ == '__main__':
    main()