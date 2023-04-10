import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import helper

def startyourengine():
	print('')
	listofwavs=[]
	cnt=0
	for i in os.listdir():
		if '.wav' in i:
			listofwavs.append(i[:-4])
			print(cnt,i)
			cnt+=1
	print('')
	print("Choose wav file: ",end='')
	x=int(input(''))
	x=listofwavs[x]
	tmp = helper.readWav(x+'.wav')
	if len(tmp)==1:
		ir = tmp
		freq, arr = helper.getFrequencyResponse(ir, normalize=True)
		plt.plot(freq, arr)
		plt.xscale('log')
		plt.show()
		reduceby = float(input('Reduce by (in dB): '))
		arr = np.clip(arr, None, -reduceby)
		ir = helper.createMinPhase(freq,arr)
		helper.writeWav(ir, f'myresultfir/{x}_{reduceby:.1f}.wav')
	if len(tmp)==2:
		ir1, ir2= tmp[0], tmp[1]
		freq, arr1 = helper.getFrequencyResponse(ir1, normalize=True)
		freq, arr2 = helper.getFrequencyResponse(ir2, normalize=True)
		plt.plot(freq, arr1, label='Left')
		plt.plot(freq, arr2, label='Right')
		plt.legend()
		plt.xscale('log')
		plt.show()
		reduceby = float(input('Reduce by (in dB): '))
		arr1 = np.clip(arr1, None, -reduceby)
		arr2 = np.clip(arr2, None, -reduceby)
		ir1, ir2 = helper.createMinPhaseStereo(freq,arr1,arr2)
		helper.writeWavStereo(ir1, ir2, f'myresultfir/{x}_{reduceby:.1f}.wav')
	

startyourengine()