#-------------------------------------------------------------------
# ccphen - Phenomenological core collapse waveforms
#
# Version: 4.0
#
# Author: Pablo Cerda-Duran. Jaime de Cabo (for the angular dependence).
#
# Year: 2020-2025
#
# File: ccphen.py
#
# Description: python3 scripts to generate the waveforms
#
#-------------------------------------------------------------------


import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_double
import math
import random

#------------------------------------------------------------
# input type for the function
# must be a double array, with single dimension that is contiguous
array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')

libccphen = npct.load_library("libccphen.so", ".")

libccphen.hphen_c.restype = c_int
libccphen.hphen_c.argtypes = [c_double, c_int, c_int, c_double, c_double, c_double,
                              c_int, array_1d_double, array_1d_double, array_1d_double, 
                              c_double, c_double, c_int, c_int, c_int,
                              array_1d_double, array_1d_double, array_1d_double, 
                              array_1d_double, array_1d_double, array_1d_double]



#===========================================================================================
def hphen_c (fs,N,par,seed, error, method):
    wfc = waveform_comp()
    h = np.zeros(N)
    ht = np.zeros(N)
    omega = np.zeros(N)
    hrms = np.zeros(N)
    forcing = np.zeros(N)
    Ntrig = np.zeros(1)
    time_pw = np.array(par.time_pw)
    f_pw = np.array(par.f_pw)
    h_pw = np.array(par.h_pw)
    if (par.use_time == 0):
        time = np.zeros(N)
    else:
        if (len(par.time) != N):
            print ('******* ERROR: wrong size of par.time *********')
            return None
        time = par.time


    ierr = libccphen.hphen_c (fs, N, seed, par.Tini, par.Tend, par.fdriver
                              , par.npw, time_pw, f_pw, h_pw
                              , par.Q, error, method, par.use_time, par.forcing_type
                              , time, h, omega, hrms, forcing, Ntrig)
    wfc.time=time
    wfc.h=h
    wfc.ht=ht
    wfc.omega=omega
    wfc.hrms=hrms
    wfc.forcing=forcing
    wfc.Ntrig=Ntrig[0]
    wfc.Q=par.Q
    return wfc


#===========================================================================================
# ---- Input parameters ------------
class param(object):
    # 0: allocate time array inside hphen. 1: pass allocated array.
    use_time = 0
    # 0: impulsive forcing. 1: white noise. 2: convective forcing
    forcing_type = 1
    # input time array (for use_time=1)
    time = 0.
    # Beginning of the forcing (post-bounce time, seconds)
    Tini = 0.
    # End of the forcing (post-bounce time, seconds)
    Tend = 0.
    # Number of triggers per second (driver frequency in Hz)
    fdriver = 100
    # Q factor for the damping
    Q = 0.

    # Number of points to define the curve
    npw=0
    # Value of time at each point
    time_pw = 0.
    # Values of omega at each point
    f_pw = 0.
    # Amplitude at each point
    h_pw = 0.

    
#===========================================================================================
# ---- Waveform component----------
class waveform_comp(object):
    # time in seconds (array)
    time = 0.
    # strain (array)
    h = 0.
    # time derivative of h in seconds^-1 (array)
    ht = 0.
    # frequency of the forcing in Hz (array)
    omega = 0.
    # instantaneous rms value of h
    hrms = 0.
    # Forcing term (array)
    forcing = 0.
    # Number of trigers
    Ntrig = 0
    # Q factor for the damping
    Q = 0.
    # All parameters used to generate the waveform
    fs=0
    N=0
    par=param()
    seed_array=[]
    dist=0.
    phi=0.
    theta=0.
    error=0.
    method=0
    log_hrms_mean=0.
    log_hrms_sigma=0.
    hrms_real=0.

#===========================================================================================
# ---- Waveform ----------
class waveform(object):
    # time in seconds (array)
    time = 0.
    # strain (array)
    h = 0.
    # dictionary of waveform components
    comp={}
    
#===========================================================================================
# ----- Function computing the waveform (two polarizations) ------
# Input:
#   fs: sampling frequency of the output
#   N: number of samples of the output
#   par: parameters of the waveform
#   seed_array: seed numbers to initialize random number generators
#               for each component (list of 5 numbers).
#   dist: distance in pc
#   phi: Observation angle phi
#   theta: Observation angle theta
#   error (optional): error required in the time integration
#   method (optional): method used for the time integration. 
#                     default=4 (fastest, more accurate)
#   log_hrms_mean: mean value of the log(hrms) at 10 kpc (see calibration below)
#   log_hrms_sigma: standard deviation of the log(hrms) at 10 kpc (calibrated)
# Output:
#   wfc: waveform component class object
# Notes:
#   Time integrator is adapted to the frequency of the 
#   oscillator. With the default parameter (error=1e-3)
#   typical error in the waveform is about 5% and it requires
#   at most 0.5 s of CPU time per second of waveform. Decreasing
#   error to 1e-4, reduces error by a factor 10 and increases the
#   CPU time by a factor 10.
#
#   Calibrated log strain at 10 kpc (mean and standart deviation)
#     - Calibration for ccphen v4 (new data, weigthed with Salpeter's IMF)
#         log_hrms_mean = -23.0;
#         log_hrms_sigma = 0.4;
#     - Calibration in Lopez et al 2021 (ccphen v3)
#         log_hrms_mean = -23.1;
#         log_hrms_sigma = 0.29;

#===========================================================================================
def hphen_pol (fs,N,par,seed_array,dist,phi,theta,error=1e-3, method=4, log_hrms_mean=-23.0, log_hrms_sigma=0.4):

    import numpy as np

    seed1=int(seed_array[0])
    seed2=int(seed_array[1])
    seed3=int(seed_array[2])
    seed4=int(seed_array[3])
    seed5=int(seed_array[4])
    random.seed(int(np.sum(seed_array)))
    
    # Weights to ensure isotropy
    w20 = 1.0
    w21 = 1.0
    w22 = 1.0
    
    # Normalize weights
    wmean = math.sqrt(w20**2+2*w21**2+2*w22**2)
    w20*=np.sqrt(4*math.pi)/wmean
    w21*=np.sqrt(4*math.pi)/wmean
    w22*=np.sqrt(4*math.pi)/wmean

    # Integrate damped harmonic oscillator equation for the different components
    # of the spin weighted spherical harmonics decomposition (real and imaginary parts)
    Reh20=hphen_c (fs,N,par,seed1,error=error, method=method)
    Reh2p1=hphen_c (fs,N,par,seed2,error=error, method=method)
    Imh2p1=hphen_c (fs,N,par,seed3,error=error, method=method)
    Reh2p2=hphen_c (fs,N,par,seed4,error=error, method=method)
    Imh2p2=hphen_c (fs,N,par,seed5,error=error, method=method)

    # Compute the complex coefficients
    h20=Reh20.h
    h2p1 = (Reh2p1.h + 1j * Imh2p1.h)/np.sqrt(2)
    h2m1 = (- (Reh2p1.h - 1j * Imh2p1.h))/np.sqrt(2)
    h2p2 = (Reh2p2.h + 1j * Imh2p2.h)/np.sqrt(2)
    h2m2 = (Reh2p2.h - 1j * Imh2p2.h)/np.sqrt(2)
    
    # Spin weighted spherical harmonics (-2_Y^2m)
    sm2y20 = 1/4.*np.sqrt(15./(2.*np.pi))*np.sin(theta)**2
    sm2y2p1 = 1/8.*np.sqrt(5/np.pi)*(2*np.sin(theta)+np.sin(2.*theta))*np.exp(+1j*phi)
    sm2y2m1 = 1/8.*np.sqrt(5/np.pi)*(2*np.sin(theta)-np.sin(2.*theta))*np.exp(-1j*phi)
    sm2y2p2 = 1/16.*np.sqrt(5/np.pi)*(3.+4*np.cos(theta)+np.cos(2.*theta))*np.exp(+1j*2*phi)
    sm2y2m2 = 1/16.*np.sqrt(5/np.pi)*(3.-4*np.cos(theta)+np.cos(2.*theta))*np.exp(-1j*2*phi)

    # Add contribution of all harmonics and normalize to have the desired relative contribution of each component 
    h= (h20*sm2y20*w20 + h2p1*sm2y2p1*w21 + h2m1*sm2y2m1*w21 + h2p2*sm2y2p2*w22 + h2m2*sm2y2m2*w22)

    # Generate hrms value of the waveform a the given distance following a normal distribution of strains.
    hrms_real = 10**random.gauss(log_hrms_mean, log_hrms_sigma) * 1e4 / dist #

    # Save waveform component 
    wfc = waveform()
    wfc.time = Reh20.time
    wfc.h = h*hrms_real
    wfc.omega =Reh20.omega
    wfc.hrms =Reh20.hrms*hrms_real # Normalize strain by using the hrms value computed above
    wfc.forcing =(Reh20.forcing*w20, Reh2p1.forcing*w21, Imh2p1.forcing*w21, Reh2p2.forcing*w22, Imh2p2.forcing*w22)
    wfc.Ntrig =Reh20.Ntrig
    wfc.Q =Reh20.Q

    # Save all parameters used to generate the waveform
    wfc.fs=fs
    wfc.N=N
    wfc.par=par
    wfc.seed_array=seed_array
    wfc.dist=dist
    wfc.phi=phi
    wfc.theta=theta
    wfc.error=error
    wfc.method=method
    wfc.log_hrms_mean=log_hrms_mean
    wfc.log_hrms_sigma=log_hrms_sigma
    
    # Save value of hrms
    wfc.hrms_real = hrms_real
    
    
    return wfc



#===========================================================================================
# Computes a phenomenological spectrogram based on the waveform component wfc.
# Input:
#    freq: array of frequencies of the spectrogram
#    time: array of times of the spectrogram
#    wfc: waveform component
# Output:
#    Sxx: Power spectral density of shape (len(freq),len(time) 
# Notes:
#    This version is vectorized for better performance and should take
#    about 4 ms for 1 second waveform at 16 kHz.
#===========================================================================================
def hphen_spect (freq, time, wfc):
    Nf = len(freq)
    Nt = len(time)
    freq0=(np.interp(time, wfc.time, wfc.omega)/(2.*math.pi)).reshape((1,Nt))
    hrms=np.interp(time, wfc.time, wfc.hrms).reshape((1,Nt))
    tmp = 2.3548200450309493 # 2.*np.sqrt(2.*np.log(2.))
    tmp2 = (wfc.Q*tmp)**2/(freq0**2*2.)
    freqs=freq.reshape((Nf,1))
    Sxx = np.exp(-(freqs-freq0)**2*tmp2)*hrms**2
    return Sxx


#===========================================================================================
# Computes rms values of a burst-like signal h(t). 
# Input:
#    time: time array
#    h: strain array
#    f: fraction of energy to be used for the computation of the duration (default 0.99)
# Output:
#    hrms: rms value of the signal
# Notes:
#    To compute the rms values the duration of the signal has to be estimated. This is done
#    considering the time interval where 99% of the signal energy is located (or a different
#    fraction set by the parameter f)
#===========================================================================================
def compute_hrms (time, h, f=0.99):
    dt = time[1]-time[0]
    # Cummulative signal energy
    Ene=np.cumsum(np.abs(h)**2)*dt
    # Normalized cummulative signal energy
    Ene=Ene/Ene[-1]
    # Compute the time interval with 99% (default value) of the signal energy 
    idx=np.where((Ene>(1-f)/2.) & (Ene<1-(1-f)/2))
    T99=np.max(time[idx]) - np.min(time[idx])
    # compute rms value of the signal
    hrms=np.sqrt(np.sum(np.abs(h[idx])**2)*dt/T99)
    return hrms


#===========================================================================================
# Generates parameters for the case of a the dominant mode in standard neutrino driven supernova.
# No Input
#
# Output:
#     theta: source inclination
#     phi: polarization angle
#     par: parameters 
#     log_hrms_mean: mean log hrms value of the dominant mode at 10 kpc  
#     log_hrms_sigma: standard deviation of the distribution of log hrms values of the dominant mode at 10 kpc  
# Notes:
#     We set a minimun duration time of 0.4 seconds. Shorter signals are treated
#     appart and usually correspond to low mass progenitors (<10 Msun) or some low
#     metallicity progenitors.
#===========================================================================================
def GenerateRandomParStandardDominantComponent():
    # Waveform calibration parameters
    log_hrms_mean=-23
    log_hrms_sigma=0.4
    # Random location in the sky
    cos_theta=np.random.uniform(-1,1)
    theta=np.arccos(cos_theta)
    phi=np.random.uniform(0,2.*math.pi)
    # waveform parameters
    par=param()
    par.npw=3
    par.time_pw = [0.,0.5,1.5]
    par.h_pw = [1.0,1.0,1.0]
    tmin=0.4 # Minimun duration of the waveforms
    for j in range(1000): # Iterate until a valid set of parameters is found 
        valid=True
        # Duration (beginning/end)
        par.Tini=np.random.uniform(0,0.25) 
        par.Tend=np.random.uniform(0.2,1.5)
        if (par.Tend-par.Tini<tmin): valid=False # too short   
        if (valid): break
    if (not valid): print ("ERROR D1")
    par.Q = np.random.uniform(1.,10.)
    f1=np.random.uniform(700,2500)
    for j in range(1000): # Iterate until a valid set of parameters is found 
        valid=True
        f0=np.random.uniform(50,150)
        f2=np.random.uniform(1500,4000)
        
        if (f0>f1): valid=False
        if (f1>f2): valid=False
        if (f0>f2): valid=False
        if ((f1-f0)/(par.time_pw[1]-par.time_pw[0])<(f2-f1)/(par.time_pw[2]-par.time_pw[1])): valid=False
        par.f_pw = [f0,f1,f2] 
        if (valid): break
    if (not valid): print ("ERROR D2")
    return theta, phi, par, log_hrms_mean, log_hrms_sigma


#===========================================================================================
# Generates parameters for the case of SASI in standard neutrino driven supernova.
# Input:
#     tmin: set lower limit for Tini (optional)
#     tmax: set upper limit to Tend (optional)   
# Output:
#     theta: source inclination
#     phi: polarization angle
#     par: parameters 
#     log_hrms_mean: mean log hrms value of the dominant mode at 10 kpc  
#     log_hrms_sigma: standard deviation of the distribution of log hrms values of the dominant mode at 10 kpc  
# Notes:
#     This is the parameter space for the SASI component of standard
#     neutrino driven supernovae. This component should be used in 
#     combination with the dominant component.
#===========================================================================================
def GenerateRandomParSASIComponent(tmin=0,tmax=1.5):
    # Waveform calibration parameters
    log_hrms_mean=-23
    log_hrms_sigma=0.4
    # Random location in the sky
    cos_theta=np.random.uniform(-1,1)
    theta=np.arccos(cos_theta)
    phi=np.random.uniform(0,2.*math.pi)
    # waveform parameters
    par=param()
    par.npw=3
    par.time_pw = [0.,1.0,1.1]
    par.h_pw = [1.0,1.0,1.0]
    
    # Duration (beginning/end)
    par.Tini=np.random.uniform(max(0,tmin),0.25) 
    par.Tend=np.random.uniform(0.2,min(1.5,tmax))
    par.Q = np.random.uniform(1.,10.)  

    f1=np.random.uniform(50,300) 
    f0=np.random.uniform(50,min(f1,150))
    f2=f1
    par.f_pw = [f0,f1,f2]
    
    return theta, phi, par, log_hrms_mean, log_hrms_sigma


#===========================================================================================
# Generates parameters for the case of a the dominant mode in short neutrino driven supernova.
# No Input
#
# Output:
#     theta: source inclination
#     phi: polarization angle
#     par: parameters 
#     log_hrms_mean: mean log hrms value of the dominant mode at 10 kpc  
#     log_hrms_sigma: standard deviation of the distribution of log hrms values of the dominant mode at 10 kpc  
# Notes:
#    This is the parameter space for short neutrino-driven supernovae that may be
#    associated to low-mass progenitors (<10 Msun) or some low metallicity 
#    progenitors. Duration is bound between 0.1 and 0.4 s. 
#===========================================================================================
def GenerateRandomParShortDominantComponent():
    # Waveform calibration parameters
    log_hrms_mean=-23
    log_hrms_sigma=0.4
    # Random location in the sky
    cos_theta=np.random.uniform(-1,1)
    theta=np.arccos(cos_theta)
    phi=np.random.uniform(0,2.*math.pi)
    # waveform parameters
    par=param()
    par.npw=3
    par.time_pw = [0.,0.5,1.5]
    par.h_pw = [1.0,1.0,1.0]
    tmin=0.1 # Minimun duration of the waveforms
    tmax=0.4 # Maximum duration of the waveforms
    # Duration (beginning/end)
    par.Tini=0.
    for j in range(10000): # Iterate until a valid set of parameters is found 
        valid=True
        par.Tend=np.random.uniform(0.2,0.4)
        if (par.Tend-par.Tini<tmin): valid=False # too short
        if (par.Tend-par.Tini>tmax): valid=False # too long
        if (valid): break
    if (not valid): print ("ERROR s1")
    par.Q = np.random.uniform(1.,10.)
    f1=np.random.uniform(700,2500)
    for j in range(10000): # Iterate until a valid set of parameters is found 
        valid=True    
        f0=np.random.uniform(50,150)
        f2=np.random.uniform(1500,4000)
        if (f0>f1): valid=False
        if (f1>f2): valid=False
        if (f0>f2): valid=False
        if ((f1-f0)/(par.time_pw[1]-par.time_pw[0])<(f2-f1)/(par.time_pw[2]-par.time_pw[1])): valid=False
        par.f_pw = [f0,f1,f2] 
        if (valid): break
    if (not valid): print ("ERROR s2")
    return theta, phi, par, log_hrms_mean, log_hrms_sigma

#===========================================================================================
# Generates a waveform corresponding to a standard (T>0.4s) neutrino-driven supernova
# without SASI component
# Input:
#   dist: distance in pc
#   fs: sampling rate (Hz)
#   N: length of the data segment
#   seed: integer number to seed random number generators
#   time (optional): array of times
# Output:
#   wf: waveform class object
#===========================================================================================
def GenerateRandomWaveformStandardNeutrinoDriven (dist, fs, N, seed, time=[0]):
    
    # time step
    dt = 1.0/fs
    # segmant duration
    T =N*dt 


    #### RNG seeds seeds ###############
    # list of 5 integers for the random number generators
    np.random.seed(seed)
    seed_array =list((np.random.random(5)*seed).astype(int))


    ### Generate rest of the parameters randomly #######
    # Generate random parameters for a Neutrino driven supernova at 10 kpc
    theta, phi, par, log_hrms_mean, log_hrms_sigma =GenerateRandomParStandardDominantComponent()

    ### Use time array on input if provided
    if len(time)>1:
        par.use_time=1
        par.time=time
    
    ### Generate waveform ###
    wfc = hphen_pol (fs,N,par,seed_array,dist,phi,theta,log_hrms_mean=log_hrms_mean, log_hrms_sigma=log_hrms_sigma)

    wf = waveform()
    wf.time=wfc.time
    wf.h=wfc.h
    wf.comp={"dominant":wfc}
    
    return wf

#===========================================================================================
# Generates a waveform corresponding to a standard (T>0.4s) neutrino-driven supernova
# with a SASI component
# Input:
#   dist: distance in pc
#   fs: sampling rate (Hz)
#   N: length of the data segment
#   seed: integer number to seed random number generators
#   time (optional): array of times
# Output:
#   wf: waveform class object
#===========================================================================================
def GenerateRandomWaveformStandardNeutrinoDrivenSASI (dist, fs, N, seed, time=[0]):
    
    # time step
    dt = 1.0/fs
    # segmant duration
    T =N*dt
    
    np.random.seed(seed)
    
    #### Dominant component ###########
    # RNG seeds seeds 
    seed_array =list((np.random.random(5)*seed).astype(int))
    # Generate rest of the parameters randomly
    theta, phi, par_dom, log_hrms_mean, log_hrms_sigma =GenerateRandomParStandardDominantComponent() 
    ### Use time array on input if provided
    if len(time)>1:
        par.use_time=1
        par.time=time
    # Generate waveform component
    wfc_dom = hphen_pol (fs,N,par_dom,seed_array,dist,phi,theta,log_hrms_mean=log_hrms_mean, log_hrms_sigma=log_hrms_sigma)

    #### SASI component ###########
    # RNG seeds seeds 
    seed_array =list((np.random.random(5)*seed).astype(int))
    # Generate rest of the parameters randomly
    theta, phi, par_sasi, log_hrms_mean, log_hrms_sigma =GenerateRandomParSASIComponent(tmin=par_dom.Tini, tmax=par_dom.Tend)
    # overwrite some of the parameters to match those of the dominant component
    theta=wfc_dom.theta
    phi=wfc_dom.phi
    log_hrms_mean=np.log10(wfc_dom.hrms_real*np.random.uniform(0.1,0.5))
    log_hrms_sigma=0.
    par_sasi.Tini=par_dom.Tini
    par_sasi.Tend=par_dom.Tend
    # Use the same time array as the dominant to generate the SASI component.
    par_sasi.use_time=True
    par_sasi.time=wfc_dom.time
    # Generate waveform component
    wfc_sasi = hphen_pol (fs,N,par_sasi,seed_array,dist,phi,theta,log_hrms_mean=log_hrms_mean, log_hrms_sigma=log_hrms_sigma)


    wf = waveform()
    wf.time=wfc_dom.time
    wf.h=wfc_dom.h + wfc_sasi.h
    wf.comp={"dominant":wfc_dom, "SASI":wfc_sasi}

    return wf

#===========================================================================================
# Generates a waveform corresponding to a short (T<0.4s) neutrino-driven supernova
# without SASI component
# Input:
#   dist: distance in pc
#   fs: sampling rate (Hz)
#   N: length of the data segment
#   seed: integer number to seed random number generators
#   time (optional): array of times
# Output:
#   wf: waveform class object
#===========================================================================================
def GenerateRandomWaveformShortNeutrinoDriven (dist, fs, N, seed, time=[0]):
    
   # time step
    dt = 1.0/fs
    # segmant duration
    T =N*dt 


    #### RNG seeds seeds ###############
    # list of 5 integers for the random number generators
    np.random.seed(seed)
    seed_array =list((np.random.random(5)*seed).astype(int))


    ### Generate rest of the parameters randomly #######
    # Generate random parameters for a Neutrino driven supernova at 10 kpc
    theta, phi, par, log_hrms_mean, log_hrms_sigma =GenerateRandomParShortDominantComponent()

    ### Use time array on input if provided
    if len(time)>1:
        par.use_time=1
        par.time=time
        
    ### Generate waveform ###
    wfc = hphen_pol (fs,N,par,seed_array,dist,phi,theta,log_hrms_mean=log_hrms_mean, log_hrms_sigma=log_hrms_sigma)

    wf = waveform()
    wf.time=wfc.time
    wf.h=wfc.h
    wf.comp={"dominant":wfc}
    
    return wf