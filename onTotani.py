#!/usr/bin/python

from optparse import OptionParser
import random
from math import pi, sin, cos, sqrt, gamma, exp
from scipy import integrate, interpolate
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

parser = OptionParser()

optdefault = "wilson-early.txt"
parser.add_option("-e", "--input_early", dest="input_early",
                  help="Name of the early input file. Default: '%s'." \
                      % (optdefault),
                  metavar="FILENAME",
                  default=optdefault)

optdefault = "wilson-late.txt"
parser.add_option("-l", "--input_late", dest="input_late",
                  help="Name of the late input file. Default: '%s'." \
                      % (optdefault),
                  metavar="FILENAME",
                  default=optdefault)

optdefault = "tmp_o_e.txt"
parser.add_option("-o", "--output", dest="output",
                  help="Name of the output file. Default: '%s'." \
                      % (optdefault),
                  metavar="FILENAME",
                  default=optdefault)

optdefault = 1.0
parser.add_option("-n", "--normalization", dest="normalization",
                  help="Normalization factor to account for neutrino oscillations. Gets set by `genevts.py`. Default: '%s'." \
                      % (optdefault),
                  metavar="NORMALIZATION",
                  default=optdefault)

optdefault = 'x'
parser.add_option("-f", "--flavor", dest="flavor",
                  help="Flavor of Neutrino input: e, eb or x. Default: '%s'." \
                      % (optdefault),
                  metavar="FLAVOR",
                  default=optdefault)

# number of oxygen nuclei in each detector
detectors = {"SuperK": 1.05e+33,
             "HyperK": 7.0e+33}
optchoices = list(detectors.keys())
optdefault = "HyperK"
parser.add_option("-d", "--detector", dest="detector",
                  help="Detector configuration. Choices: %s. Default: %s" \
                      % (optchoices, optdefault),
                  choices=optchoices, default=optdefault)

parser.add_option("-v", "--verbose", dest="verbose",
                  help="Verbose output, e.g. for debugging. Off by default.",
                  default=False, action="store_true")

(options, args) = parser.parse_args()

verbose = options.verbose
normalization = float(options.normalization)
flavor = options.flavor
print(flavor)
if (normalization <= 0 or normalization > 1):
	print("Error: Normalization factor should be in the interval (0,1]. Aborting ...")
	exit()

# return direction of a positron with the given energy
def direction(energy):
	eneNu = energy + eThr
	pMax = 0
	cosT = 0
	nCosTBins = 1000
	cosTBinWidth = 2./nCosTBins
	for j in range(nCosTBins):
		cosT = -1 + cosTBinWidth*(j+0.5) # 1000 steps in the interval [-1,1]
		p = dir_nuebar_p_sv(eneNu, cosT)
		if p > pMax:
			pMax = p
	
	while (True):
		cosT = 2*np.random.random() - 1 # randomly distributed in interval [-1,1)
		if dir_nuebar_p_sv(eneNu, cosT) > pMax*np.random.random():
			sinT = sin(np.arccos(cosT))
			phi = 2 * pi * np.random.random() - pi # randomly distributed in [-pi, pi)
			break
	
	return (sinT*cos(phi), sinT*sin(phi), cosT)

# angular distribution for the angle at which the positron is emitted
def dir_nuebar_p_sv(eneNu, cosT):
	return 1 - ((1+((eneNu - eThr)/25)**4)/(3+((eneNu - eThr)/25)**4))*cosT

nevtValues=[]
tVal=[]
EeVal=[]
EebVal=[]
ExVal=[]
totnevt=0
tVal_bin = [0]
#Define variables
#number of oxygen nuclei in chosen detector
nO = detectors[options.detector]
dSquared = (1.563738e+33)**2
eThr = 15 #MeV

#open the early totani file
with open(options.input_early) as f: lines = f.read().splitlines()
if verbose: print ("Reading neutrino simulation data from", options.input, "...")

#define a section of the code that collects all the relevant info in that section and is callable for any line in the code
def section(num):
    EVal=[0]
    fVal=[]
    hVal=[]
    hmVal=[]
    speceVal=[0]
    specebVal=[0]
    specxVal=[0]
    EeVal_bin=[0]
    EebVal_bin=[0]
    ExVal_bin=[0]
    end = num + 42
    for line in lines[num:end]:
        if line == lines[num]:
            t,x,y,z,v,w,s,q = line.split() #only relevant value is t, other letters are irrelevent 
            t=float(t)
            tVal.append(t*1000)
            tVal_bin.append(t)
        if line == lines[num+3]:
            Ee_bin,Eeb_bin,Ex_bin,E_out = line.split()
            Ee_bin=float(Ee_bin)
            Eeb_bin=float(Eeb_bin)
            Ex_bin=float(Ex_bin)/4 #divide by four as only want one of the Nu_x species and Totani files contain total of all 4
            EeVal_bin.append(Ee_bin)
            EebVal_bin.append(Eeb_bin)
            ExVal_bin.append(Ex_bin)
        if line == lines[num+16]:
            Ee,Eeb,Ex = line.split()
            Ee = float(Ee)/1000
            Eeb = float(Eeb)/1000
            Ex = float(Ex)/1000
            EeVal.append(Ee)
            EebVal.append(Eeb)
            ExVal.append(Ex)
        for j in range(19,39):
            if line == lines[num+j]:
                n,E,fo,ho,hmo,fn,hn,hmn,f,h,hm=line.split()
                E,f,h,hm=float(E)/1000,float(f),float(h),float(hm) #f is nu_e, h is nu_eb,hm is nu_x
                EVal.append(E)
                fVal.append(f)
                hVal.append(h)
                hmVal.append(hm)
    #luminosity calculated for each energy bin by doing change in energy over change in time
    Le = (EeVal_bin[-1]-EeVal_bin[-2])/(tVal_bin[-1]-tVal_bin[-2])
    Leb = (EebVal_bin[-1]-EebVal_bin[-2])/(tVal_bin[-1]-tVal_bin[-2])
    Lx = (ExVal_bin[-1]-ExVal_bin[-2])/(tVal_bin[-1]-tVal_bin[-2])

    E_int_e = 0
    E_int_eb = 0
    E_int_x = 0

    #calculate energy spectrum direct from totani files
    for i in range(len(fVal)-1):
        spec_e = fVal[i]/(EVal[i+2]-EVal[i])
        spec_eb = hVal[i]/(EVal[i+2]-EVal[i])
        spec_x = hmVal[i]/(EVal[i+2]-EVal[i])
        
        speceVal.append(spec_e)
        specebVal.append(spec_eb)
        specxVal.append(spec_x)
        
        E_int_e = E_int_e + (speceVal[i]+speceVal[i+1])*(EVal[i+1]-EVal[i])*0.5
        E_int_eb = E_int_eb + (specebVal[i]+specebVal[i+1])*(EVal[i+1]-EVal[i])*0.5
        E_int_x = E_int_x + (specxVal[i]+specxVal[i+1])*(EVal[i+1]-EVal[i])*0.5

    spec_eVal = np.array(speceVal)/E_int_e
    spec_ebVal = np.array(specebVal)/E_int_eb
    spec_xVal = np.array(specxVal)/E_int_x


    #If you want to use the 22.5-100 fit (the original Super-K fit) for lower energies, simply comment out all but the final return statement
    def sigma(eNu):
        #if eNu < 22.5:
        #   return (2.767e-45*(eNu**0.1815-15**0.1815)**3.387)*2.56819e25
        #else:
            return (4.7e-44*(eNu**0.25-15**0.25)**6)*2.56819e25


    if flavor == 'e':
        def dFluxdE():
                return 1/(4*pi*dSquared)*((Le*624.151)/Ee)*spec_eVal

    if flavor == 'eb':
        def dFluxdE():
                return 1/(4*pi*dSquared)*((Leb*624.151)/Eeb)*spec_ebVal
        
    if flavor == 'x':
        def dFluxdE():
                return 1/(4*pi*dSquared)*((Lx*624.151)/Ex)*spec_xVal
    
    #interpolate dFluxdE over energy bins 
    intF = interpolate.pchip(EVal[1:],dFluxdE())

        
    #integrate over eNu to obtain the event rate at time t
    def f(eNu):
        return sigma(eNu)*intF(eNu)
    
    #calculate the detector event rate at time t by integrating from eThr to 50MeV
    simnevt = nO  * integrate.quad(f, eThr, 50) [0]
             
    #create a list of nevt values at time (t) for input into interpolation function
    nevtValues.append(simnevt)
        



#call every section in wilson-early.txt
section(0)
for line, i in zip(lines,range(len(lines))):
    if line.startswith('----'): #doesn't work for original wilson-early file, but replace '----' in this line with '-' or '--' and it should be fine
        section(i+2)
             
with open(options.input_late) as f: lines = f.read().splitlines()
def sectionLate(num):
    end = num + 45
    EVal=[0]
    fVal=[]
    hVal=[]
    hmVal=[]
    speceVal=[0]
    specebVal=[0]
    specxVal=[0]
    EeVal_bin=[0]
    EebVal_bin=[0]
    ExVal_bin=[0]
    for line in lines[num:end]:
        if line == lines[num]:
            t = line.split()
            t=float(t[0])
            tVal.append(t*1000)
            tVal_bin.append(t)
        if line == lines[num+5]:
            Ee,Eeb,Ex,E_out = line.split()
            Ee=float(Ee)
            Eeb=float(Eeb)
            Ex=float(Ex)/4 #divide by four as only want one of the Nu_x species and Totani files contain total of all 4
            EeVal_bin.append(Ee)
            EebVal_bin.append(Eeb)
            ExVal_bin.append(Ex)
        if line == lines[num+18]:
            Ee,Eeb,Ex = line.split()
            Ee = float(Ee)/1000
            Eeb = float(Eeb)/1000
            Ex = float(Ex)/1000
            EeVal.append(Ee)
            EebVal.append(Eeb)
            ExVal.append(Ex)
        for j in range(21,40):
            if line == lines[num+j]:
                n,E,f,h,hm=line.split()
                E,f,h,hm=float(E)/1000,float(f),float(h),float(hm) #f is nu_e, h is nu_eb,hm is nu_x
                EVal.append(E)
                fVal.append(f)
                hVal.append(h)
                hmVal.append(hm)
    #luminosity calculated for each energy bin by doing change in energy over change in time
    Le = (EeVal_bin[-1]-EeVal_bin[-2])/(tVal_bin[-1]-tVal_bin[-2])
    Leb = (EebVal_bin[-1]-EebVal_bin[-2])/(tVal_bin[-1]-tVal_bin[-2])
    Lx = (ExVal_bin[-1]-ExVal_bin[-2])/(tVal_bin[-1]-tVal_bin[-2])

    E_int_e = 0
    E_int_eb = 0
    E_int_x = 0
    
    #calculate energy spectrum direct from totani files
    for i in range(len(fVal)-1):

        spec_e = fVal[i]/(EVal[i+2]-EVal[i])
        spec_eb = hVal[i]/(EVal[i+2]-EVal[i])
        spec_x = hmVal[i]/(EVal[i+2]-EVal[i])
        
        speceVal.append(spec_e)
        specebVal.append(spec_eb)
        specxVal.append(spec_x)
        
        E_int_e = E_int_e + (speceVal[i]+speceVal[i+1])*(EVal[i+1]-EVal[i])*0.5
        E_int_eb = E_int_eb + (specebVal[i]+specebVal[i+1])*(EVal[i+1]-EVal[i])*0.5
        E_int_x = E_int_x + (specxVal[i]+specxVal[i+1])*(EVal[i+1]-EVal[i])*0.5

    spec_eVal = np.array(speceVal)/E_int_e
    spec_ebVal = np.array(specebVal)/E_int_eb
    spec_xVal = np.array(specxVal)/E_int_x
    

    #If you want to use the 22.5-100 fit (the original Super-K fit) for lower energies, simply comment out all but the final return statement
    def sigma(eNu):
        #if eNu < 22.5:
        #   return 2.767e-45*(eNu**0.1815-15**0.1815)**3.387
        #else:
            return (4.7e-44*(eNu**0.25-15**0.25)**6)*2.56819e25

    
        
    if flavor == 'e':
        def dFluxdE():
                return 1/(4*pi*dSquared)*((Le*624.151)/Ee)*spec_eVal

    if flavor == 'eb':
        def dFluxdE():
                return 1/(4*pi*dSquared)*((Leb*624.151)/Eeb)*spec_ebVal
        
    if flavor == 'x':
        def dFluxdE():
                return 1/(4*pi*dSquared)*((Lx*624.151)/Ex)*spec_xVal

    #interpolate dFluxdE over energy bins 
    intF = interpolate.pchip(EVal[1:],dFluxdE())

        
    #integrate over eNu to obtain the event rate at time t
    def f(eNu):
        return sigma(eNu)*intF(eNu)
    
    #calculate the detector event rate at time t by integrating from eThr to 50MeV
    simnevt = nO  * integrate.quad(f, eThr, 50) [0]
             
    #create a list of nevt values at time (t) for input into interpolation function
    nevtValues.append(simnevt)
        


#call every section in wilson-late.txt
for line, i in zip(lines,range(len(lines))):
    if line.startswith(' ----'):
        sectionLate(i-2)
             

#interpolate the mean energy

interpolatedEnergy = interpolate.pchip(tVal, ExVal)

#interpolate the event rate            
interpolatedNevt = interpolate.pchip(tVal, nevtValues)
#specify bin width and number of bins for binning to 1ms intervals
binWidth = 1 #time interval in ms
binNr = np.arange(1, 535/binWidth, 1) #time range

outfile = open(options.output, 'w')
#integrate event rate and energy over each bin
for i in binNr:
        time = 15 + (i*binWidth)
        boundsMin = time - binWidth
        boundsMax = time

        # calculate expected number of events in this bin and multiply with a factor
        # (1, sin^2(theta_12), cos^2(theta_12)) to take neutrino oscillations into account
        binnedNevt = integrate.quad(interpolatedNevt, boundsMin, boundsMax)[0] * normalization
        #print(binnedNevt)
        # randomly select number of events in this bin from Poisson distribution around binnedNevt:
        binnedNevtRnd = np.random.choice(np.random.poisson(binnedNevt, size=1000))
        
        #find the total number of events over all bins
        totnevt += binnedNevtRnd
        
        binnedEnergy = integrate.quad(interpolatedEnergy, boundsMin, boundsMax)[0]
        
        if verbose:
                print ("**************************************")
                print ("timebin       = %s-%s ms" % (boundsMin, boundsMax))
                print ("Nevt (theor.) =", binnedNevt)
                print ("Nevt (actual) =", binnedNevtRnd)
                print ("mean energy   =", binnedEnergy, "MeV")
                print ("Now generating events for this bin ...")

        #define particle for each event in time interval
        for i in range(binnedNevtRnd):
                #Define properties of the particle
                 t = time - np.random.random()
                 #ene = np.random.gamma(alpha+1, binnedEnergy/(alpha+1))
                 #(dirx, diry, dirz) = direction(ene)
        
                 # print out [t, pid, energy, dirx, diry, dirz] to file
                 #outfile.write("%f, -11, %f, %f, %f, %f\n" % (t, ene, dirx, diry, dirz))

print ("**************************************")
print(("Wrote %i particles to " % totnevt) + options.output)

outfile.close()
