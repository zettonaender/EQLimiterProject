import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fftp
import scipy.interpolate as interpolate
import csv
import os
import shutil
import sys
from autoeq import batch_processing
showplot=False

def rp(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def doeq():
	batch_processing(input_dir="ssweep",output_dir=rp('myresult/ssweep'),standardize_input=True,compensation='zero.csv',equalize=True,show_plot=showplot,convolution_eq=True,fs=[48000])
	print('===============================================')
	print('Graphic EQ generated. Check -> Output Folder/'+"ssweep")

def startyourengine():
	x=input("Input file name: ")
	sr,datastr=wavfile.read(rp(x+'.wav'))
	data=datastr[:,0]
	datar=datastr[:,1]

	#Left Channel
	arr=fftp.rfft(data,sr)
	freq=fftp.rfftfreq(len(arr)*2,1/sr)
	arr=20*np.log10(abs(arr))
	freq=freq[:-1].copy()
	leftamp=arr
	#plt.plot(freq,leftamp)

	#Right Channel
	arr=fftp.rfft(datar,sr)
	freq=fftp.rfftfreq(len(arr)*2,1/sr)
	arr=20*np.log10(abs(arr))
	freq=freq[:-1].copy()
	rightamp=arr
	#plt.plot(freq,rightamp)

	#Get maxamp
	mxleft=0.0
	for i in range(len(leftamp)):
		mxleft=max(mxleft,leftamp[i])
	mxright=0.0
	for i in range(len(rightamp)):
		mxright=max(mxright,rightamp[i])

	#Do reduction
	reduceby=float(input("Enter reduction in dB: "))
	clampleft=mxleft-reduceby
	clampright=mxright-reduceby
	for i in range(len(leftamp)):
		leftamp[i]=min(leftamp[i],clampleft)
	for i in range(len(rightamp)):
		rightamp[i]=min(rightamp[i],clampright)
	#plt.plot(freq,leftamp)
	#plt.plot(freq,rightamp)

	#Debug
	#print("Yea")
	#plt.xscale("log")
	#plt.show()

	freqarr=[]
	with open(rp('zero.csv')) as f:
		reader=csv.reader(f,delimiter=',')
		for i in reader:
			freqarr.append(i[0])

	tmp=interpolate.interp1d(freq,leftamp)
	arrnew=tmp(freqarr)
	arrnew=-1*arrnew
	with open(rp('left.csv'), mode='w', newline='') as output:
		writer=csv.writer(output,delimiter=',')
		writer.writerow(['frequency','raw'])
		for i in range(0,len(freqarr)):
			writer.writerow([str(freqarr[i]),str(round(arrnew[i],2))])

	tmp=interpolate.interp1d(freq,rightamp)
	arrnew=tmp(freqarr)
	arrnew=-1*arrnew
	with open(rp('right.csv'), mode='w', newline='') as output:
		writer=csv.writer(output,delimiter=',')
		writer.writerow(['frequency','raw'])
		for i in range(0,len(freqarr)):
			writer.writerow([str(freqarr[i]),str(round(arrnew[i],2))])

	try:
		os.mkdir(rp('ssweep/'))
	except OSError:
		print ('')
	shutil.move(rp('left.csv'),rp('ssweep/left.csv'))
	shutil.move(rp('right.csv'),rp('ssweep/right.csv'))

	doeq()

startyourengine()