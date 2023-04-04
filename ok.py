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

def doeq(x,tmp):
	batch_processing(input_dir="ssweep",output_dir=rp('myresult/'+x),standardize_input=True,compensation='zero.csv',equalize=True,show_plot=showplot,convolution_eq=True,fs=[48000],parametric_eq=tmp,max_filters=[100,100])
	print('')
	print('Generated. Check -> '+'myresult/'+x)

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
	sr,datastr=wavfile.read(rp(x+'.wav'))
	data=datastr[:,0]
	datar=datastr[:,1]
	if data.dtype==np.int16:
		data=data/32768
		datar=datar/32768
	elif data.dtype==np.int32:
		data=data/2147483648
		datar=datar/2147483648

	#Left Channel
	arr=fftp.rfft(data)
	freq=fftp.rfftfreq(len(data),1/sr)
	arr=20*np.log10(abs(arr))
	leftamp=arr
	leftamp-=np.max(leftamp)
	plt.plot(freq,leftamp,label='left')

	#Right Channel
	arr=fftp.rfft(datar)
	freq=fftp.rfftfreq(len(datar),1/sr)
	arr=20*np.log10(abs(arr))
	rightamp=arr
	rightamp-=np.max(rightamp)
	plt.plot(freq,rightamp,label='right')

	#Plot
	plt.xscale('log')
	plt.legend()
	plt.show()

	#Get maxamp
	mxleft=0.0
	for i in range(len(leftamp)):
		mxleft=max(mxleft,leftamp[i])
	mxright=0.0
	for i in range(len(rightamp)):
		mxright=max(mxright,rightamp[i])

	#Do reduction
	reduceby=float(input("Enter reduction in dB: "))
	x+='_'+str(reduceby)
	mx=max(mxleft,mxright)
	clampleft=mx-reduceby
	clampright=mx-reduceby
	for i in range(len(leftamp)):
		leftamp[i]=min(leftamp[i],clampleft)
	for i in range(len(rightamp)):
		rightamp[i]=min(rightamp[i],clampright)

	freqarr=[]
	with open(rp('zero.csv')) as f:
		reader=csv.reader(f,delimiter=',')
		for i in reader:
			freqarr.append(i[0])

	tmp=interpolate.interp1d(freq,leftamp)
	arrnew=tmp(freqarr)
	arrnew=-1*arrnew
	with open(rp(x+'left.csv'), mode='w', newline='') as output:
		writer=csv.writer(output,delimiter=',')
		writer.writerow(['frequency','raw'])
		for i in range(0,len(freqarr)):
			writer.writerow([str(freqarr[i]),str(round(arrnew[i],2))])

	tmp=interpolate.interp1d(freq,rightamp)
	arrnew=tmp(freqarr)
	arrnew=-1*arrnew
	with open(rp(x+'right.csv'), mode='w', newline='') as output:
		writer=csv.writer(output,delimiter=',')
		writer.writerow(['frequency','raw'])
		for i in range(0,len(freqarr)):
			writer.writerow([str(freqarr[i]),str(round(arrnew[i],2))])

	try:
		os.mkdir(rp('ssweep/'))
	except OSError:
		pass
	shutil.move(rp(x+'left.csv'),rp('ssweep/'+x+' left.csv'))
	shutil.move(rp(x+'right.csv'),rp('ssweep/'+x+' right.csv'))

	print('Generate Parametric EQ? (y/n): ', end='')
	tmp=input('')
	print('')
	if tmp=='y':
		tmp=True
	else:
		tmp=False

	try:
		shutil.rmtree(rp('myresult/'+x))
	except FileNotFoundError:
		pass

	doeq(x,tmp)

	try:
		shutil.rmtree('ssweep')
	except FileNotFoundError:
		pass

	sr,leftwav=wavfile.read(rp('myresult/'+x+'/'+x+' left minimum phase 48000Hz.wav'))
	sr,rightwav=wavfile.read(rp('myresult/'+x+'/'+x+' right minimum phase 48000Hz.wav'))
	outputpath=rp('myresult/'+x+'/'+x+'.wav')
	data=np.vstack((leftwav[:,0],rightwav[:,0]))
	data=data.transpose()
	wavfile.write(outputpath,sr,data)

	if tmp:
		leftparam=rp('myresult/'+x+'/'+x+' left ParametricEQ.txt')
		rightparam=rp('myresult/'+x+'/'+x+' right ParametricEQ.txt')
		outputtext=[]
		outputtext.append('Preamp: 0 dB\nChannel: L\nPreamp: 0 dB\nChannel: R\nPreamp: 0 dB\nChannel: L\n')
		with open(leftparam,'r') as f:
			for i in f.readlines():
				outputtext.append(i)
			f.close()
		outputtext.append('\nChannel: R\n')
		with open(rightparam,'r') as f:
			for i in f.readlines():
				outputtext.append(i)
			f.close()
		outputparam=rp('myresult/'+x+'/'+x+' ParametricEQ.txt')
		with open(outputparam,'w') as f:
			for i in outputtext:
				f.write(i)
			f.close()

	print('\n')

startyourengine()
