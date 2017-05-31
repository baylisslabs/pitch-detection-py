import numpy as np
import matplotlib.pyplot as plt
import math
import random

f0 = 220.0;

semi_tones = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

class params:
    sampleRate = 44100.0
    windowLength = 1200;

def midi_num_to_note(note_number):
    octave = int(note_number) / 12
    semi_tone = int(note_number) % 12
    return "%s%d" % (semi_tones[semi_tone],octave-1)

def midi_note_even_temp(f):
    return 69.0+12*math.log(f/440.0)/math.log(2)

def prob_density(data,a,b,n_bins):
    pdf = np.zeros(n_bins)
    bin_size = (b-a)/n_bins
    for x in data:
        bin_no = int((x-a)/bin_size)
        if bin_no < 0:
            bin_no = 0
        if bin_no >= n_bins:
            bin_no = n_bins-1
        pdf[bin_no] = pdf[bin_no] + 1
    return pdf

def CDF(pdf_data):
    cdf = integrate(pdf_data)
    return cdf

def get_zero_slope_events(data):
    diff = differentiate(data)
    events = []
    a = None
    for i in range(1,len(data)):
        b = diff[i-1]  
        if b == 0.0 and a is None:
            events.append((i,data[i]))           
        elif a is not None and b <= 0.0 and a > 0.0:
            events.append((i,data[i]))
        elif a is not None and b >= 0.0 and a < 0.0:
            events.append((i,data[i]))
        a = b
    return events

def slope_event_data(events):
    y = []
    for ev in events:
        y.append(ev[1])
    return y


def downsample(a,s,data):
    y = []
    n = (len(data)-a)/s
    for i in range(0,n):
        y.append(data[a+i*s])
    return y

def mean(data):   
    sum = 0
    for i in data:
        sum += i4mr
    return sum/len(data)


def ppa_single(s,data):
    max_v = 0
    for i in range(0,s):
        m = (mean(downsample(i,s,data)))
        m = m*m
        if m > max_v:
            max_v = m
    return max_v

def PPA(points,data):
    y = []
    for x in points:
        y.append(ppa_single(x,data))
    return y

def acf_single(s,data):
    b = len(data)-s
    sum = 0
    for i in range(0,b):
        diff = data[i]-data[i+s]
        sum += data[i]*data[i+s]
    return sum

def acf_single(s,data):
    b = len(data)-s
    sum = 0
    for i in range(0,b):
        diff = data[i]-data[i+s]
        sum += data[i]*data[i+s]
    return sum

def mss_single(s,data):
    b = len(data)-s
    sum = 0
    for i in range(0,b):
        ss = data[i]*data[i]+data[i+s]*data[i+s]
        sum += ss
    return sum

def nsdf_single(s,data):
    return 2.0*acf_single(s,data)/mss_single(s,data)
            

def sdf_single(s,data):
    b = len(data)-s
    sum = 0
    for i in range(0,b):
        diff = data[i]-data[i+s]
        sum += diff*diff
    return sum / b

def dft_single(s,data):
    sum = complex(0,0)        
    for n in range(0,len(data)):
        a = - 2.0 * math.pi * n / s
        re = data[n] * math.cos(a)
        im = data[n] * math.sin(a)        
        sum += complex(re,im)
    return sum

def DFT(points,data):
    y = []
    for x in points:
        y.append(dft_single(x,data))
    return y

def SDF(points,data):
    y = []
    for x in points:
        y.append(sdf_single(x,data))
    return y

def ACF(points,data):
    y = []
    for x in points:
        y.append(acf_single(x,data))
    return y

def NSDF(points,data):
    y = []
    for x in points:
        y.append(nsdf_single(x,data))
    return y

def getSineWave(points,sampleRate,freq,amp,phase):
    y = []
    for x in points:        
        y.append(amp*math.sin(phase+2.0*math.pi*x*freq/sampleRate))
    return y


def getAmpDensity(complex_data):
    y = []
    for x in complex_data:
        y.append(abs(x))
    return y

def getPwrSigned(data,p):
    y = []
    for x in data:
        x2 = x**p
        y.append(-x2 if x<0 else x2)
    return y

def getPwr(data,p):
    y = []
    for x in data:
        x2 = x**p
        y.append(x2)
    return y

def getExpSigned(data):
    y = []
    for x in data:
        x2 = math.exp(x)
        y.append(-x2 if x<0 else x2)
    return y


def getExp(data):
    y = []
    for x in data:
        x2 = math.exp(x)
        y.append(x2)
    return y

def getLog(data):
    y = []
    for x in data:
        x2 = math.log(x+1)
        y.append(x2)
    return y

def getInv(data):
    y = []
    for x in data:
        x2 = 1.0/x
        y.append(x2)
    return y

def norm(data,a,b):
    y = []
    max_x = -1e32
    min_x = 1e32
    amp = b - a
    for x in data:        
        if max_x<x:
            max_x = x
        if min_x>x:
            min_x = x
    scale = (max_x-min_x)
    if scale > 0:
        for i in range(0,len(data)):
            y.append(a+(data[i]-min_x)/scale*amp)
        return y
    else:
        return data

def gate(data,thresh):
    y = []
    for x in data: 
        l = 0
        if abs(x) > thresh:
            l=x
        y.append(l)
    return y

def latch(data,thresh):
    y = []
    l = 0
    for x in data:        
        if x > thresh:
            l=1.0
        elif x < -thresh:
            l=-1.0
        y.append(l)
    return y

def differentiate(data):
    y = []
    a = data[0]
    for i in range(1,len(data)):
        b = data[i]
        y.append(b-a)
        a = b
    return y

def integrate(data):
    y = []
    a = 0
    for i in range(0,len(data)):
        b = data[i]+a
        y.append(b)
        a = b
    return y

def array_abs(data):
    return [abs(x) for x in data]

def lowpass(data,a):    
    n = len(data)
    y = []
    alpha = a;
    y.append(data[0])
    for i in range(1,n):
        y.append(y[i-1]+alpha*(data[i]-y[i-1]))    
    return y

def map_fft_to_period_space(points,fft_data,sampleRate):
    bin_size = sampleRate/len(fft_data)
    y = []
    for x in points:        
        freq = sampleRate/x
        bin_num = freq / bin_size
        if bin_num<len(fft_data):
            y.append(fft_data[bin_num])
        else:
            y.append(0)
    return y  
    
mn0 = midi_note_even_temp(f0)
name0 = midi_num_to_note(mn0)
print "f0: ", f0, "p0: ", params.sampleRate/f0 , "Note:", mn0, " = ", name0

P0 = int(params.sampleRate/f0)
F0 = params.sampleRate/P0
F01 = params.sampleRate/(P0+1.0)
print "F0", F0, "P0: ", P0 , "Note:", midi_note_even_temp(F0)
print "F01", F01, "P0: ", P0+1 , "Note:", midi_note_even_temp(F01)
random.seed()
x = range(0,params.windowLength)
y0 = getSineWave(x,params.sampleRate,f0,1.0,0)
y1 = getSineWave(x,params.sampleRate,2*f0,2.0,0)
y2 = getSineWave(x,params.sampleRate,16*f0,0.0,0.0)
#y2 = [random.uniform(-1.0,1.0) for i in x]
#y = lowpass(lowpass(agc(np.add(np.add(y0,y1),y2),1.0),0.005),0.005);
y = latch(norm(np.add(np.add(y0,y1),y2),-1.0,1.0),0.9);
iy= y# norm(integrate(y),-1.0,1.0)
#sev = get_zero_slope_events(iy)
#sed = slope_event_data(sev)
#sex = range(0,len(sev))
#sdf_sed = agc(SDF(sex,sed),1.0)
#pdf = prob_density(array_abs(sed),0.0,1.0,1000)
#cdf = norm(CDF(pdf),0.0,1.0)
#pdfx = range(0,len(pdf)) 
yn = (y) #agc(lowpass(y,1.0,1e32,1),1.0)
#w = np.hanning(len(y))
#ye = getExpSigned(yn)
#yen = agc(ye,1.0)
sdf_x = range(1,params.windowLength/2+1)
sdf_y = norm(SDF(sdf_x,iy),0.0,1.0)
sdf_y_1 = [1.0-t for t in sdf_y]
#sdf_y = NSDF(sdf_x,yn)
#ppa_y = PPA(sdf_x,yn)
#fft_y = np.abs(np.fft.fft(y))
#fft_ps = map_fft_to_period_space(sdf_x,fft_y,params.sampleRate)
#dft_y = DFT(sdf_x,np.multiply(yn,w))
#dft_pd = agc((getAmpDensity(dft_y)),1.0)

# Create the plot
plt.figure(1)
plt.subplot(211)
plt.plot(x,iy)
#plt.subplot(512)
#plt.plot(sex,sed)
#plt.subplot(513)
#plt.bar(pdfx,[x+1e-7 for x in pdf])
#plt.subplot(514)
#plt.bar(pdfx,[x+1e-7 for x in cdf])
plt.subplot(212);
plt.plot(sdf_x,sdf_y)

# Save the figure in a separate file
#plt.savefig('sine_function_plain.png')

# Draw the plot to the screen
plt.show()

