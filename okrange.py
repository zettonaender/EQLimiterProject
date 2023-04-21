import matplotlib.pyplot as plt
import numpy as np
import os
import helper


def getMaxInRange(freq, arr, minfreq, maxfreq):
    mx = np.min(arr)
    for i in range(len(freq)):
        if freq[i] >= minfreq and freq[i] <= maxfreq:
            mx = max(mx, arr[i])
    return mx


def startyourengine():
    print('')
    listofwavs = []
    cnt = 0
    for i in os.listdir():
        if '.wav' in i:
            listofwavs.append(i[:-4])
            print(cnt, i)
            cnt += 1
    print('')
    print("Choose wav file: ", end='')
    x = int(input(''))
    x = listofwavs[x]
    startfreq = float(input('Start frequency: '))
    endfreq = float(input('End frequency: '))

    tmp = helper.readWav(x+'.wav')
    if len(tmp) == 1:
        ir = tmp
        freq, arr = helper.getFrequencyResponse(ir, normalize=True)
        arr -= np.max(arr)
        reduceby = getMaxInRange(freq, arr, startfreq, endfreq)
        arr = np.clip(arr, None, reduceby)
        arr -= 0.1
        ir = helper.createMinPhase(freq, arr, taplength=131072)
        helper.writeWav(ir, f'myresultfir/{x}_{startfreq:.1f}_{endfreq:.1f}.wav')
    if len(tmp) == 2:
        ir1, ir2 = tmp[0], tmp[1]
        freq, arr1 = helper.getFrequencyResponse(ir1, normalize=False)
        freq, arr2 = helper.getFrequencyResponse(ir2, normalize=False)
        arr1 -= np.max(arr1)
        arr2 -= np.max(arr2)
        max1 = getMaxInRange(freq, arr1, startfreq, endfreq)
        max2 = getMaxInRange(freq, arr2, startfreq, endfreq)
        reduceby = max(max1, max2)
        arr1 = np.clip(arr1, None, reduceby)
        arr2 = np.clip(arr2, None, reduceby)
        arr1 -= 0.1
        arr2 -= 0.1
        ir1, ir2 = helper.createMinPhaseStereo(
            freq, arr1, arr2, taplength=131072)
        helper.writeWavStereo(ir1, ir2, f'myresultfir/{x}_{startfreq:.1f}_{endfreq:.1f}.wav')


startyourengine()
