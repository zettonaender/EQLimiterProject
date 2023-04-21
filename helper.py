import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import scipy.fft as fft
import soundfile as sf
import csv


def plotImpulseResponse(ir, name=None, sr=48000):
    freq = fft.rfftfreq(len(ir), 1/sr)
    arr = 20*(np.log10(np.abs(fft.rfft(ir))))
    if name is None:
        plt.plot(freq, arr)
    else:
        plt.plot(freq, arr, label=name)


def plotFrequencyResponse(freq, arr, name=None):
    if name is None:
        plt.plot(freq, arr)
    else:
        plt.plot(freq, arr, label=name)


def showPlot():
    plt.legend()
    plt.xscale('log')
    plt.show()


def readCSV(filename):
    # Read frequency response CSVs
    freq = []
    arr = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                freq.append(float(row[0]))
                arr.append(float(row[1]))
            # REW
            except IndexError:
                freq = []
            # AutoEQ
            except ValueError:
                pass
    freq = np.array(freq)
    arr = np.array(arr)
    return freq, arr


def createMinPhase(freq, arr, length=-1, taplength=16384, sr=48000, normalize='fft'):
    if normalize == 'fft':
        arr -= np.max(arr) + 0.1

    # Preprocess for firwin2
    if freq[0] > 0:
        freq = np.insert(freq, 0, 0)
        arr = np.insert(arr, 0, arr[0])
    if freq[-1] < sr/2:
        freq = np.append(freq, sr/2)
        arr = np.append(arr, arr[len(arr)-1])
    arr *= 2
    arr = 10**(arr/20)

    # Create filter
    if taplength % 2 == 0:
        taplength += 1
    h = signal.firwin2(taplength, freq, arr, fs=sr)
    h = signal.minimum_phase(h)

    # Windowing
    if length > 0:
        lengthInSamples = length*2*sr
        window = signal.windows.hann(lengthInSamples)
        window = window[lengthInSamples//2:]
        h = h[:lengthInSamples//2]
        h *= window

    # Normalize
    if normalize == 'ir':
        h /= np.max(h)

    return h


def createMinPhaseStereo(freq, arr1, arr2, length=-1, taplength=16384, sr=48000, normalize='fft'):
    if normalize == 'fft':
        arr1 -= np.max(arr1) + 0.1
        arr2 -= np.max(arr2) + 0.1

    # Preprocess for firwin2
    if freq[0] > 0:
        freq = np.insert(freq, 0, 0)
        arr1 = np.insert(arr1, 0, arr1[0])
        arr2 = np.insert(arr2, 0, arr2[0])
    if freq[-1] < sr/2:
        freq = np.append(freq, sr/2)
        arr1 = np.append(arr1, arr1[len(arr1)-1])
        arr2 = np.append(arr2, arr2[len(arr2)-1])
    arr1 *= 2
    arr1 = 10**(arr1/20)
    arr2 *= 2
    arr2 = 10**(arr2/20)

    # Create filter
    if taplength % 2 == 0:
        taplength += 1
    h1 = signal.firwin2(taplength, freq, arr1, fs=sr)
    h2 = signal.firwin2(taplength, freq, arr2, fs=sr)
    h1 = signal.minimum_phase(h1)
    h2 = signal.minimum_phase(h2)

    # Windowing
    if length > 0:
        lengthInSamples = length*2*sr
        window = signal.windows.hann(lengthInSamples)
        window = window[lengthInSamples//2:]
        h1 = h1[:lengthInSamples//2]
        h2 = h2[:lengthInSamples//2]
        h1 *= window
        h2 *= window

    # Normalize
    if normalize == 'ir':
        h1 /= np.max(h1)
        h2 /= np.max(h2)

    return h1, h2


def getFrequencyResponse(ir, sr=48000, linear=False, normalize=False):
    freq = fft.rfftfreq(len(ir), 1/sr)
    tmp = np.abs(fft.rfft(ir))
    if linear:
        return freq, tmp
    tmp[tmp == 0] = 0.0001
    arr = 20*(np.log10(tmp))
    if normalize:
        arr -= np.max(arr)
    return freq, arr


def getDB(arr):
    arr[arr == 0] = 0.0001
    arr = 20*(np.log10(arr))
    return arr


def getLinear(arr):
    arr = 10**(arr/20)
    return arr


def readWav(filename, Normalize=True):
    sr, ir = wavfile.read(filename)
    if ir.dtype == np.int16:
        ir = ir.astype(np.float32)
        ir = ir / 32767
    if ir.dtype == np.int32:
        ir = ir.astype(np.float32)
        ir = ir / 2147483647
    elif ir.dtype != np.float32 and ir.dtype != np.float64:
        print(ir.dtype)
        raise ValueError('Unsupported type.')
    if Normalize:
        ir /= np.max(np.abs(ir))
    try:
        channelCount = ir.shape[1]
    except IndexError:
        channelCount = 1
    if channelCount == 2:
        return [ir[:, 0], ir[:, 1]]
    return [ir]


def writeWav(ir, filename, type='float32', sr=48000):
    if type == 'int16':
        ir *= 32767
        ir = ir.astype(np.int16)
    wavfile.write(filename, sr, ir)


def writeWavStereo(ir1, ir2, filename, type='float32', sr=48000):
    if type == 'int16':
        ir1 *= 32767
        ir1 = ir1.astype(np.int16)
        ir2 *= 32767
        ir2 = ir2.astype(np.int16)
    ir = np.transpose(np.vstack((ir1, ir2)))
    wavfile.write(filename, sr, ir)


def convolveArr(input, filter, normalize=True):
    h = signal.convolve(input, filter, mode='full')
    if normalize:
        h /= np.max(h)
    return h


def resizeIRLikeREW(h, sr=48000, normalize=True):
    # Find index of abs peak
    peakIndex = np.argmax(np.abs(h))

    # Split into two arrays, the peak is in the second array
    h1 = h[:peakIndex]
    h2 = h[peakIndex:]

    # Create window
    leftWindowLength = int(sr*0.125)
    rightWindowLength = int(sr*0.5)

    leftWindow = signal.windows.hann(leftWindowLength*2)[:leftWindowLength]
    rightWindow = signal.windows.hann(rightWindowLength*2)[rightWindowLength:]

    # Zero pad the arrays if they are too short
    if len(h1) < leftWindowLength:
        h1 = np.insert(h1, 0, np.zeros(leftWindowLength-len(h1)))
    else:
        h1 = h1[(len(h1)-leftWindowLength):]
    if len(h2) < rightWindowLength:
        h2 = np.append(h2, np.zeros(rightWindowLength-len(h2)))
    else:
        h2 = h2[:rightWindowLength]

    # Apply window
    h1 *= leftWindow
    h2 *= rightWindow

    # Concatenate
    h = np.append(h1, h2)

    # Normalize
    if normalize:
        h /= np.max(h)

    return h


def writeFLAC(ir, filename, sr=48000):
    ir *= 32767
    ir = ir.astype(np.int16)
    sf.write(filename, ir, sr)


def readFLAC(filename):
    ir, sr = sf.read(filename)
    ir = ir.astype(np.float32)
    ir /= 32767
    ir /= np.max(np.abs(ir))
    return ir
