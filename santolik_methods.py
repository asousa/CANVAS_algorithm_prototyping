import numpy as np
from scipy.fftpack import fft
import scipy
from scipy.signal.windows import hann
from scipy.signal import blackman
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import butter, lfilter



Q_EL = 1.602e-19
M_EL = 9.1e-31
eo   = 8.854e-12
B0   = 30e-6
VC    = 2.998e8



# Plasma helper methods 
def gen_rotation_matrix(v1, v2):
    ''' generates a rotation matrix to map v1 onto v2'''
    
    
    a,b = v1/np.linalg.norm(v1), v2/np.linalg.norm(v2)
    c = np.dot(a,b)
    if c== 1:
        return np.diag(np.ones_like(v1))
    else:
        v = np.cross(a,b)
        s = np.linalg.norm(v)
        vXStr = '{} {} {}; {} {} {}; {} {} {}'.format(0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0)
        k = np.matrix(vXStr)
        r = np.diag(np.ones(len(v1))) + k + k@k * ((1 -c)/(s**2))

        return r


def ne_ps(L, doy, Kp):
#     The Carpenter + Anderson equatorial plasmasphere density
#     profile (Gallagher 2000, equation 5).

    a6 = -0.79
    a7 = 5.208
    a8 = 5.39 - 0.382*Kp;  # Plasmapause location in L (moldwin fit)
    a9 = 0.5  # Slope of plasmapause transition in L (I think)
    rz12 = 0  # mean sunspot number
    
#     ! ---------- Carpenter / Anderson plasmasphere -------------
    h = (1.0 + (L/a8)**(2.0*(a9 - 1)))**(-a9/(a9 - 1.0))
    doy_factor=np.pi*(doy+9.0)/365.0
    x234=( 0.1*(np.cos(2.0*doy_factor) - 0.5*np.cos(4.0*doy_factor))+ (0.00127*rz12-0.0635) ) * np.exp(-(L-2.0)/1.5)

    ne_ps = 10**(a6*L+a7 + x234)

    return ne_ps*1e6

## Appleton-Hartree solution:
def appleton_hartree(Bmag, Ne, theta, phi, f):
    k_vec_AH   = np.zeros(2)
    eta_vec_AH =np.zeros(2)
    
    Wc   = Q_EL*Bmag/M_EL
    Wc2  = pow(Wc,2)
    Wp2  = Ne*pow(Q_EL,2)/eo/M_EL
    Wp   = np.sqrt(Wp2)
#     print(f)
    w = 2.*np.pi*f
#     print(w)
    numerator = Wp2/w/w
    denom1 = (Wc2*pow(np.sin(phi),2))/2./(w*w - Wp2)
    denom2 = np.sqrt(pow(denom1,2) + Wc2*pow(np.cos(phi), 2)/w/w)
#     print(denom1, denom2, numerator)
    eta2_1   = (1 - numerator/(1 - denom1 + denom2))
    eta2_2   = (1 - numerator/(1 - denom1 - denom2))
    eta_vec_AH[0] = np.sqrt(eta2_1)
    eta_vec_AH[1] = np.sqrt(eta2_2)

    k_vec_AH = eta_vec_AH*w/VC
    
    
    return eta_vec_AH, k_vec_AH

def get_fft(sig, fs, N):
    T = 1.0/fs
    if N is None:
        N = len(sig)
        
    w = blackman(N)
    yf = fft(sig[:N]*w)
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2.)
    
    return xf, yf

def gen_td(Lshell, freq, theta, phi, pol_angle, E_mag, tmax, fs):

    Bmag = B0/pow(Lshell - 1, 3)
    ne = ne_ps(Lshell, 1, 0)


    tvec = np.arange(0,tmax,1./fs)


    ex = np.zeros_like(tvec)
    ey = np.zeros_like(tvec)
    ez = np.zeros_like(tvec)
    bx = np.zeros_like(tvec)
    by = np.zeros_like(tvec)
    bz = np.zeros_like(tvec)


    
    eta_per_freq, k_per_freq = appleton_hartree(Bmag, ne, theta, phi, freq)
    
    if np.all(np.isnan(k_per_freq)):
#         print("invalid input parameters")
        return None, None, None, None, None, None
    else:
        kmag = k_per_freq[~np.isnan(k_per_freq)][0]

    kdir = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

    # Simulate some time-domain data!

    w = freq*Hz2Rad

    jones_vec = np.array([np.exp(1j*pol_angle), np.exp(1j*pol_angle)*1j,0])

    # Rotate into same frame as propagation direction:
    M = gen_rotation_matrix([0,0,1],kdir)

    jones_vec = np.array(np.dot(M, jones_vec), 'complex')
    mag_vec = np.array(E_mag*jones_vec,'complex').ravel()

    exl = np.array(np.real(mag_vec[0]*np.exp(1j*w*tvec - kdir[0])),'float')
    eyl = np.array(np.real(mag_vec[1]*np.exp(1j*w*tvec - kdir[1])),'float')
    ezl = np.array(np.real(mag_vec[2]*np.exp(1j*w*tvec - kdir[2])),'float')

    kvec = kmag*kdir

    bxl = (kvec[1]*ezl - kvec[2]*eyl)/VC
    byl = (kvec[2]*exl - kvec[0]*ezl)/VC
    bzl = (kvec[0]*eyl - kvec[1]*exl)/VC

    ex += exl
    ey += eyl
    ez += ezl
    bx += bxl
    by += byl
    bz += bzl

    return ex,ey,ez,bx,by,bz


def add_noise(ex,ey,ez,bx,by,bz, e_noise_mag, b_noise_mag):

    ex+=np.random.randn(len(ex))*e_noise_mag
    ey+=np.random.randn(len(ey))*e_noise_mag
    ez+=np.random.randn(len(ez))*e_noise_mag

    bx+=np.random.randn(len(ex))*b_noise_mag
    by+=np.random.randn(len(ey))*b_noise_mag
    bz+=np.random.randn(len(ez))*b_noise_mag 
    
# ex,ey,ez,bx,by,bz = gen_td(3,2000, 45*D2R, 0, 0, 1, 0.1, fs)
# print(ex)


def plot_td():
    tvec = np.arange(0,tmax,1./fs)
    fig, ax = plt.subplots(3,2, sharex=True, sharey='col')
    ax[0,0].plot(tvec, ex)
    ax[1,0].plot(tvec, ey)
    ax[2,0].plot(tvec, ez)
    ax[0,1].plot(tvec, bx)
    ax[1,1].plot(tvec, by)
    ax[2,1].plot(tvec, bz)
    fig.autofmt_xdate()
    fig.tight_layout()
    ax[0,0].set_title('E')
    ax[0,1].set_title('B')
    return fig

def plot_fd():
    
    freqs, FEx = get_fft(ex, fs, N)
    freqs, FEy = get_fft(ey, fs, N)
    freqs, FEz = get_fft(ez, fs, N)
    freqs, FBx = get_fft(bx, fs, N)
    freqs, FBy = get_fft(by, fs, N)
    freqs, FBz = get_fft(bz, fs, N)
    
    fig, ax = plt.subplots(3,2)
    ax[0,0].plot(freqs,20*np.log10(np.abs(FEx[0:N//2])))
    ax[1,0].plot(freqs,20*np.log10(np.abs(FEy[0:N//2])))
    ax[2,0].plot(freqs,20*np.log10(np.abs(FEz[0:N//2])))

    ax[0,1].plot(freqs,20*np.log10(np.abs(FBx[0:N//2])))
    ax[1,1].plot(freqs,20*np.log10(np.abs(FBy[0:N//2])))
    ax[2,1].plot(freqs,20*np.log10(np.abs(FBz[0:N//2])))

    return fig

def plot_hists():
    fig, ax = plt.subplots(2,1)
    h1 = ax[0].hist(err['theta'], 200)
    ax[0].set_title(r'Error in $\theta$')
    ax[0].set_xlabel(r'Error in $\theta$ [deg]')
    ax[0].set_ylabel('counts')
#     ax[0].set_xlim([-5,5])
    # ax[0].set_xlim([-90,90])
    h2 = ax[1].hist(err['phi'], 200)
    ax[1].set_title(r'Error in $\phi$')
    ax[1].set_xlabel(r'Error in $\phi$ [deg]')
    ax[1].set_ylabel('counts')
    fig.suptitle('hey betch')
#     ax[1].set_xlim([-5,5])
    return fig




# Plasma helper methods 
def gen_rotation_matrix(v1, v2):
    ''' generates a rotation matrix to map v1 onto v2'''
    
    
    a,b = v1/np.linalg.norm(v1), v2/np.linalg.norm(v2)
    c = np.dot(a,b)
    if c== 1:
        return np.diag(np.ones_like(v1))
    else:
        v = np.cross(a,b)
        s = np.linalg.norm(v)
        vXStr = '{} {} {}; {} {} {}; {} {} {}'.format(0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0)
        k = np.matrix(vXStr)
        r = np.diag(np.ones(len(v1))) + k + k@k * ((1 -c)/(s**2))

        return r


def ne_ps(L, doy, Kp):
#     The Carpenter + Anderson equatorial plasmasphere density
#     profile (Gallagher 2000, equation 5).

    a6 = -0.79
    a7 = 5.208
    a8 = 5.39 - 0.382*Kp;  # Plasmapause location in L (moldwin fit)
    a9 = 0.5  # Slope of plasmapause transition in L (I think)
    rz12 = 0  # mean sunspot number
    
#     ! ---------- Carpenter / Anderson plasmasphere -------------
    h = (1.0 + (L/a8)**(2.0*(a9 - 1)))**(-a9/(a9 - 1.0))
    doy_factor=np.pi*(doy+9.0)/365.0
    x234=( 0.1*(np.cos(2.0*doy_factor) - 0.5*np.cos(4.0*doy_factor))+ (0.00127*rz12-0.0635) ) * np.exp(-(L-2.0)/1.5)

    ne_ps = 10**(a6*L+a7 + x234)

    return ne_ps*1e6

## Appleton-Hartree solution:
def appleton_hartree(Bmag, Ne, theta, phi, f):
    k_vec_AH   = np.zeros(2)
    eta_vec_AH =np.zeros(2)
    
    Wc   = Q_EL*Bmag/M_EL
    Wc2  = pow(Wc,2)
    Wp2  = Ne*pow(Q_EL,2)/eo/M_EL
    Wp   = np.sqrt(Wp2)
#     print(f)
    w = 2.*np.pi*f
#     print(w)
    numerator = Wp2/w/w
    denom1 = (Wc2*pow(np.sin(phi),2))/2./(w*w - Wp2)
    denom2 = np.sqrt(pow(denom1,2) + Wc2*pow(np.cos(phi), 2)/w/w)
#     print(denom1, denom2, numerator)
    eta2_1   = (1 - numerator/(1 - denom1 + denom2))
    eta2_2   = (1 - numerator/(1 - denom1 - denom2))
    eta_vec_AH[0] = np.sqrt(eta2_1)
    eta_vec_AH[1] = np.sqrt(eta2_2)

    k_vec_AH = eta_vec_AH*w/VC
    
    
    return eta_vec_AH, k_vec_AH

def get_fft(sig, fs, N):
    T = 1.0/fs
    if N is None:
        N = len(sig)
        
    w = blackman(N)
    yf = fft(sig[:N]*w)
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2.)
    
    return xf, yf

def gen_td_tone(Lshell, freq, theta, phi, pol_angle, E_mag, tmax, fs):
    ''' Generate a set of time-domain signals, which contain a monochromatic Whistler wave.
        Specified by 
            Location (L-shell)
            requency (Hz)
            K vector angles theta and phi (radians)
            pol_angle (Polarization angle, radians)
            E_mag (Magnitude of the E field, arbitrary units -- mV/meter is customary)
            tmax (maximum time, in seconds)
            fs (sampling frequency, in hz)
         '''

    # Get the background magnetic field, as a function of L-shell
    Bmag = B0/pow(Lshell - 1, 3)

    # Get the plasmasphere electron density, using Carpenter and Anderson
    ne = ne_ps(Lshell, 1, 0)

    tvec = np.arange(0,tmax,1./fs)

    # Initialize E and B vectors
    ex = np.zeros_like(tvec)
    ey = np.zeros_like(tvec)
    ez = np.zeros_like(tvec)
    bx = np.zeros_like(tvec)
    by = np.zeros_like(tvec)
    bz = np.zeros_like(tvec)

    # Solve for Eta and K at this location and frequency
    # (This could be made fancier, using Stix, as in the raytracer -- but this is fine here)
    eta_per_freq, k_per_freq = appleton_hartree(Bmag, ne, theta, phi, freq)
    
    if np.all(np.isnan(k_per_freq)):
#         print("invalid input parameters")
        return None, None, None, None, None, None
    else:
        kmag = k_per_freq[~np.isnan(k_per_freq)][0]

    # Get K direction as a unit vector
    kdir = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

    # Simulate some time-domain data!
    w = freq*Hz2Rad

    jones_vec = np.array([np.exp(1j*pol_angle), np.exp(1j*pol_angle)*1j,0])

    # Rotate into same frame as propagation direction:
    M = gen_rotation_matrix([0,0,1],kdir)

    jones_vec = np.array(np.dot(M, jones_vec), 'complex')
    mag_vec = np.array(E_mag*jones_vec,'complex').ravel()

    exl = np.array(np.real(mag_vec[0]*np.exp(1j*w*tvec - kdir[0])),'float')
    eyl = np.array(np.real(mag_vec[1]*np.exp(1j*w*tvec - kdir[1])),'float')
    ezl = np.array(np.real(mag_vec[2]*np.exp(1j*w*tvec - kdir[2])),'float')

    # K vector has magnitude Kmag, and direction Kdir
    kvec = kmag*kdir

    # B field is related to E via k 
    bxl = (kvec[1]*ezl - kvec[2]*eyl)/VC
    byl = (kvec[2]*exl - kvec[0]*ezl)/VC
    bzl = (kvec[0]*eyl - kvec[1]*exl)/VC

    ex += exl
    ey += eyl
    ez += ezl
    bx += bxl
    by += byl
    bz += bzl

    return ex,ey,ez,bx,by,bz


def add_noise(ex,ey,ez,bx,by,bz, e_noise_mag, b_noise_mag):
    ''' Add some Gaussian noise to E and B '''

    ex+=np.random.randn(len(ex))*e_noise_mag
    ey+=np.random.randn(len(ey))*e_noise_mag
    ez+=np.random.randn(len(ez))*e_noise_mag

    bx+=np.random.randn(len(ex))*b_noise_mag
    by+=np.random.randn(len(ey))*b_noise_mag
    bz+=np.random.randn(len(ez))*b_noise_mag 
    
# ex,ey,ez,bx,by,bz = gen_td(3,2000, 45*D2R, 0, 0, 1, 0.1, fs)
# print(ex)

def plot_spec(ex, ey, ez, bx, by, bz, fs,  E_clims = [-12, -4], B_clims = [-20, -15], nfft=512, overlap=0.5, window='hanning', shading='gouorad'):
    ''' Plot spectrograms of E and B '''
    # fig, ax = plt.subplots(4,2,sharex=True,sharey=True)
    fig, ax = plt.subplots(3,2,sharex=True,sharey=True)

    c = 2.998e8
    # Get spectra
    ff,tt, FBx = scipy.signal.spectrogram(bx, fs=fs, window=window, nperseg=nfft, noverlap=nfft*overlap,mode='complex')
    ff,tt, FBy = scipy.signal.spectrogram(by, fs=fs, window=window, nperseg=nfft, noverlap=nfft*overlap,mode='complex')
    ff,tt, FBz = scipy.signal.spectrogram(bz, fs=fs, window=window, nperseg=nfft, noverlap=nfft*overlap,mode='complex')

    ff,tt, FEx = scipy.signal.spectrogram(1e-3*ex, fs=fs, window=window, nperseg=nfft, noverlap=nfft*overlap,mode='complex')
    ff,tt, FEy = scipy.signal.spectrogram(1e-3*ey, fs=fs, window=window, nperseg=nfft, noverlap=nfft*overlap,mode='complex')
    ff,tt, FEz = scipy.signal.spectrogram(1e-3*ez, fs=fs, window=window, nperseg=nfft, noverlap=nfft*overlap,mode='complex')


    # # Recreate Ez, because it's noisy and it sucks

    # ReEz = -1*(np.real(FEx)*np.real(FBx) + np.real(FEy)*np.real(FBy))/np.real(FBz)
    # ImEz = -1*(np.imag(FEx)*np.imag(FBx) + np.imag(FEy)*np.imag(FBy))/np.imag(FBz)

    # if np.sum(np.isnan(ReEz))>0:
    #     print('filling real')
    #     fill_nans(tt,ff,ReEz)
    # if np.sum(np.isnan(ImEz))>0:
    #     print('filling imag')
    #     fill_nans(tt,ff,ImEz)

    # # newEz = ReEz + 1j*ImEz

    S_mag = np.sqrt(np.real(FEx*np.conj(FEx)))
    # print(np.min(S_mag), np.max(S_mag))
    p = ax[0,0].pcolormesh(tt,ff, np.log10(S_mag), cmap = plt.cm.jet,vmin=E_clims[0],vmax=E_clims[1],shading=shading)
    S_mag = np.sqrt(np.real(FEy*np.conj(FEy)))
    # print(np.min(S_mag), np.max(S_mag))
    p = ax[1,0].pcolormesh(tt,ff, np.log10(S_mag), cmap = plt.cm.jet,vmin=E_clims[0],vmax=E_clims[1],shading=shading)
    # S_mag = np.real(newEz*np.conj(newEz))
    S_mag = np.sqrt(np.real(FEz*np.conj(FEz)))
    # print(np.min(S_mag), np.max(S_mag))
    p = ax[2,0].pcolormesh(tt,ff, np.log10(S_mag), cmap = plt.cm.jet,vmin=E_clims[0],vmax=E_clims[1],shading=shading)

    # S_mag = np.sqrt(np.real(newEz*np.conj(newEz)))    
    # print(np.min(S_mag), np.max(S_mag))
    # p = ax[3,0].pcolormesh(tt,ff, np.log10(S_mag), cmap = plt.cm.jet,vmin=E_clims[0],vmax=E_clims[1],shading=shading)
    # p = ax[3,1].pcolormesh(tt,ff, np.isnan(S_mag), cmap = plt.cm.jet)





    S_mag = np.sqrt(np.real(FBx*np.conj(FBx)))
    # print(np.min(S_mag), np.max(S_mag))
    p = ax[0,1].pcolormesh(tt,ff, np.log10(S_mag), cmap = plt.cm.jet,vmin=B_clims[0],vmax=B_clims[1],shading=shading)
    S_mag = np.sqrt(np.real(FBy*np.conj(FBy)))
    # print(np.min(S_mag), np.max(S_mag))
    p = ax[1,1].pcolormesh(tt,ff, np.log10(S_mag), cmap = plt.cm.jet,vmin=B_clims[0],vmax=B_clims[1],shading=shading)
    S_mag = np.sqrt(np.real(FBz*np.conj(FBz)))
    # print(np.min(S_mag), np.max(S_mag))
    p = ax[2,1].pcolormesh(tt,ff, np.log10(S_mag), cmap = plt.cm.jet,vmin=B_clims[0],vmax=B_clims[1],shading=shading)
    # ax[0,0].set_yscale('log')
    
    ax[0,0].set_ylim([50,8000])
    ax[0,0].set_title('E')
    ax[0,1].set_title('B')
    ax[0,0].set_ylabel('X\nfrequency [hz]')
    ax[1,0].set_ylabel('Y\nfrequency [hz]')
    ax[2,0].set_ylabel('Z\nfrequency [hz]')
    ax[2,0].set_xlabel('Time [sec]')

    return fig

def fill_nans(x,y,array):
    ''' Fill any NaNs in a 2d array, using a cubic spline interpolator '''
    interp=interpolate.interp2d(x,y[1:-1],array[1:-1,:],kind='cubic',fill_value=None)
    newvals = interp(x,y[0])
    array[0,:] = newvals
    newvals = interp(x,y[-1])
    array[-1,:] = newvals

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def gen_td_whistler(L, D, theta, phi, pol_angle, tmax, fs, E_mag = 1, e_noise_mag = 1e-3, b_noise_mag = 1e-15, lowcut=600, highcut=20000):
    ''' Generates a time-domain whistler wave
        Lshell: for B-field and plasma density model
        D: Whistler dispersion coefficient
        theta, phi: Wavenormal angles, spherical coords
        pol_angle: polarization angle
        tmax: Maximium time duration, in seconds
        fs: sampling frequency
        
        Returns:
        ex, ey, ez, bx, by, bz
    '''

    D2R = np.pi/180.
    R2D = 180./np.pi
    Hz2Rad = 2.*np.pi
    Rad2Hz = 1./Hz2Rad


    Q_EL = 1.602e-19
    M_EL = 9.1e-31
    eo   = 8.854e-12
    B0   = 30e-6
    c    = 2.998e8
    mu   = 4.0*np.pi*10e-7
    B0 = 30e-6
    Bmag = B0/pow(L - 1, 3)
    ne = ne_ps(L, 1, 0)


    tvec = np.arange(0,tmax,1./fs)
    f0 = fs/2 # Max frequency

    # bandpass filter for the generated data
    filt_order = 3

    wvec= np.zeros_like(tvec)
    exl = np.zeros_like(tvec)
    eyl = np.zeros_like(tvec)
    ezl = np.zeros_like(tvec)
    bxl = np.zeros_like(tvec)
    byl = np.zeros_like(tvec)
    bzl = np.zeros_like(tvec)

    for ind, t in enumerate(tvec):
        # Whistler frequency as a function of time
        f = pow(D/t,2)
#         f = f0*(1 - t/2/tvec[-1])
        if not np.isnan(f) and (f < fs/2):
            # Get K from the plasma model
            eta_v, k_v = appleton_hartree(Bmag, ne, theta, phi, f)
            
            if np.all(np.isnan(k_v)):
                # No valid solution for this case -- skip it
                print(f'evanescent mode at {f} hz')
            else:
                # Valid frequency and K magnitude
                kmag = k_v[~np.isnan(k_v)][0]
                kdir = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

                w = f*Hz2Rad
                wvec[ind] = w
                jones_vec = np.array([np.exp(1j*pol_angle), np.exp(1j*pol_angle)*1j,0])

                M = gen_rotation_matrix([0,0,1],kdir)
                jones_vec = np.array(np.dot(M, jones_vec), 'complex')
                mag_vec = np.array(E_mag*jones_vec,'complex').ravel()
    
                exl[ind] = np.real(mag_vec[0]*np.exp(1j*w*t - kdir[0]))
                eyl[ind] = np.real(mag_vec[1]*np.exp(1j*w*t - kdir[1]))
                ezl[ind] = np.real(mag_vec[2]*np.exp(1j*w*t - kdir[2]))

                # K vector has magnitude Kmag, and direction Kdir
                kvec = kmag*kdir

                # B field is related to E via... E x B = w/k I think?
                bxl[ind] = (kvec[1]*ezl[ind] - kvec[2]*eyl[ind])/c
                byl[ind] = (kvec[2]*exl[ind] - kvec[0]*ezl[ind])/c
                bzl[ind] = (kvec[0]*eyl[ind] - kvec[1]*exl[ind])/c
    
    # Filter the generated whistler:
    exl = butter_bandpass_filter(exl, lowcut, highcut, fs, order=filt_order)
    eyl = butter_bandpass_filter(eyl, lowcut, highcut, fs, order=filt_order)
    ezl = butter_bandpass_filter(ezl, lowcut, highcut, fs, order=filt_order)
    bxl = butter_bandpass_filter(bxl, lowcut, highcut, fs, order=filt_order)
    byl = butter_bandpass_filter(byl, lowcut, highcut, fs, order=filt_order)
    bzl = butter_bandpass_filter(bzl, lowcut, highcut, fs, order=filt_order)

    # Let's add in some noise
    exl+= np.random.rand(len(exl))*e_noise_mag
    eyl+= np.random.rand(len(eyl))*e_noise_mag
    ezl+= np.random.rand(len(ezl))*e_noise_mag

    bxl+= np.random.rand(len(exl))*b_noise_mag
    byl+= np.random.rand(len(eyl))*b_noise_mag
    bzl+= np.random.rand(len(ezl))*b_noise_mag 

    return exl, eyl, ezl, bxl, byl, bzl

def santolik(ex,ey,ez,bx,by,bz,fs,N):
    ''' The Santolik method, for time-domain signals E and B '''
    freqs, FEx = get_fft(ex, fs, N)
    freqs, FEy = get_fft(ey, fs, N)
    freqs, FEz = get_fft(ez, fs, N)
    freqs, FBx = get_fft(bx, fs, N)
    freqs, FBy = get_fft(by, fs, N)
    freqs, FBz = get_fft(bz, fs, N)

    # output space
    theta_vec = np.zeros_like(freqs)
    phi_vec = np.zeros_like(freqs)
    n_vec = np.zeros([len(freqs), 3])
    planarity=np.zeros_like(freqs)
    Qmag = np.zeros_like(freqs)
    
    # The algorithm!
    for fi in np.arange(1,len(freqs)):
#         print('freq',freqs[fi])
        z = np.array([VC*FBx[fi], VC*FBy[fi], VC*FBz[fi], FEx[fi], FEy[fi], FEz[fi]],'complex')
        Q = np.outer(z,np.conj(z))
        
        A = []
        B = []
        for l in range(6):
            A_tmp = np.zeros([3,3],'complex')
            B_tmp = np.zeros([3],'complex')

            # Valid entries in Levi-Civita symbol (minus 1, because python is zero-indexed)
            for e,i,j,k in ([1,0,1,2],[1,1,2,0],[1,2,0,1],[-1,0,2,1],[-1,1,0,2],[-1,2,1,0]):
                A_tmp[i,j] = e*Q[k+3,l]
                B_tmp[i]= Q[i,l]
            A.extend(A_tmp)
            B.extend(B_tmp)
        # print(np.array(A))
        # print(np.array(B))
        A2 = np.concatenate([np.real(A),np.imag(A)],axis=0)
        B2 = np.concatenate([np.real(B),np.imag(B)],axis=0)

        U,s,VT=np.linalg.svd(A2,full_matrices=False)
        V = VT.T
        W_inv = np.linalg.inv(np.diag(s))
        n = np.linalg.multi_dot([V,W_inv,U.T,B2])
        
        # angles
        theta_calc = np.arctan(np.sqrt(n[0]**2 + n[1]**2)/n[2])
#         print("theta is:",theta*R2D,"deg")
        phi_calc   = np.arctan(n[1]/n[0])
        if (n[0]<0 and n[2] < 0):
            phi_calc -= np.pi
        if (n[0]<0 and n[2] >= 0):
            phi_calc += np.pi
            
            
        # wrap phi
        if phi_calc > np.pi:
            phi_calc -= 2.*np.pi
        if phi_calc <= -np.pi:
            phi_calc+= 2.*np.pi

        theta_vec[fi] = theta_calc
        phi_vec[fi] = phi_calc
        n_vec[fi,:] = n
        
        # Planarity:
        beta = A2.dot(n) 
        big_N = np.sum(pow(beta - B2,2))
        big_D = np.sum(pow(np.abs(beta) + np.abs(B2),2))
        planarity[fi] = 1 - np.sqrt(big_N/big_D)
        
        # Uh... spectral power or something? I'm making this up
        Qmag[fi] = np.sum(np.trace(Q))
    return freqs, theta_vec, phi_vec, n_vec, planarity, Qmag




def santolik_Q(Q):
    ''' The Santolik method, for Q (a 6x6 cross-spectral matrix) '''

    def wrap(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    try:
        # Assemble A and B
        A = []
        B = []
        for l in range(6):
            A_tmp = np.zeros([3,3],'complex')
            B_tmp = np.zeros([3],'complex')

            # Valid entries in Levi-Civita symbol (minus 1, because python is zero-indexed)
            for e,i,j,k in ([1,0,1,2],[1,1,2,0],[1,2,0,1],[-1,0,2,1],[-1,1,0,2],[-1,2,1,0]):
                A_tmp[i,j] = e*Q[k+3,l]
                B_tmp[i]= Q[i,l]
            A.extend(A_tmp)
            B.extend(B_tmp)

        A2 = np.concatenate([np.real(A),np.imag(A)],axis=0)
        B2 = np.concatenate([np.real(B),np.imag(B)],axis=0)

        # Solve for n using SVD
        U,s,VT=np.linalg.svd(A2,full_matrices=False)
        V = VT.T
        W_inv = np.linalg.inv(np.diag(s))
        n = np.linalg.multi_dot([V,W_inv,U.T,B2])

        # angles
        theta = np.arctan(np.sqrt(n[0]**2 + n[1]**2)/n[2])
        phi   = np.arctan(n[1]/n[0])
        
        theta = wrap(theta)
        phi = wrap(phi) 

        nmag = np.linalg.norm(n)

        # Planarity:
        beta = A2.dot(n)
        big_N = np.sum(pow(beta - B2,2))
        big_D = np.sum(pow(np.abs(beta) + np.abs(B2),2))
        planarity = 1 - np.sqrt(big_N/big_D)

        return theta, phi, planarity
    except:
        return 0, 0, 0
