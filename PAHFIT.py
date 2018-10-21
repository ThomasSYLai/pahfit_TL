# -*- coding: utf-8 -*-
# Main module for PAHFIT

from __future__ import print_function

import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import pandas as pd

from astropy.modeling.fitting import LevMarLSQFitter
from scipy import interpolate
from astropy.modeling import (models, Model, Fittable1DModel,
                              Parameter, InputParameterError)
from matplotlib.ticker import ScalarFormatter

__all__ = ['S07', 'LS18', 'bb', 'sl', 'dust', 'line']


def drude(in_lam, cen_wave, inten, frac_fwhm):
	return inten*frac_fwhm**2 / ((in_lam/cen_wave-cen_wave/in_lam)**2 + frac_fwhm**2)

def blackbody(in_x, temp):
    """
    Taking into accout the emissivity normalized at lambda = 9.7 um
    """
    return ((9.7/in_x)**2)*3.97289e13/in_x**3/(np.exp(1.4387752e4/in_x/temp)-1.)

def gaussian(in_lam, cen_wave, inten, fwhm):
    return inten * np.exp(-((in_lam-cen_wave)**2)*2.7725887222397811 / (fwhm*cen_wave)**2)

class S07(Fittable1DModel):
    """
    S07 kvt extinction model calculation

    Parameters
    ----------
    kvt_amp : float
      amplitude

    Notes
    -----
    S07 extinction model

    From Kemper, Vriend, & Tielens (2004)

    Applicable for Mid-Infrared

    A custom extinction curve constructed by two components:
    silicate profile & exponent 1.7 power-law.

    Example showing a S07 curve with components identified.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.dust_extinction import S07

    """
    inputs = ('x',)
    outputs = ('y',)#('axav',)

    # Attenuation tau
    Tau_si = Parameter(description="kvt term: amplitude", default=0.01, min=0.0, max=10)

    @staticmethod
    def kvt(in_x):
        """
        Output the kvt extinction curve
        """
        kvt_wav = np.array([8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.7,
            9.75, 9.8, 10.0, 10.2, 10.4, 10.6,10.8, 11.0, 11.2, 11.4, 
            11.6, 11.8, 12.0, 12.2, 12.4, 12.6, 12.7])

        kvt_int = np.array([.06, .09, .16, .275, .415, .575, .755, .895, .98, .99, 
            1.0, .99, .94, .83, .745, .655, .58, .525, .43, .35, 
            .27, .20, .13, .09, .06, .045, .04314])

        # Extend kvt profile to shorter wavelengths
        kvt_wav_short = in_x[in_x < min(kvt_wav)]
        kvt_int_short_tmp = min(kvt_int) * np.exp(2.03*(kvt_wav_short-min(kvt_wav)))
        # Since kvt_int_shoft_tmp does not reach min(kvt_int), we scale it to stitch it.
        kvt_int_short = kvt_int_short_tmp * (kvt_int[0]/max(kvt_int_short_tmp))

        # Extend kvt profile to longer wavelengths
        kvt_wav_long = in_x[in_x > max(kvt_wav)]
        kvt_int_long = np.zeros(len(kvt_wav_long)) # Need to input the value instead of using zeros

        spline_x = np.concatenate([kvt_wav_short, kvt_wav, kvt_wav_long])
        spline_y = np.concatenate([kvt_int_short, kvt_int, kvt_int_long])

        spline_rep = interpolate.splrep(spline_x, spline_y)
        new_spline_y = interpolate.splev(in_x, spline_rep, der=0)

        ext = drude(in_x, 18, 0.4, 0.247) + new_spline_y

        # Extend to ~2 um
        # assuing beta is 0.1 
        beta = 0.1
        y = (1.0-beta)*ext + beta*(9.7/in_x)**1.7

        return y

    def evaluate(self, in_x, Tau_si):
        ext = np.exp(-Tau_si * self.kvt(in_x))

        return ext


class LS18(Fittable1DModel):
    """
    Update extinction model from S07.
    Add ice features in AKARI spectrum: H20, CO2, CO

    Parameters
    ----------
    amp_si : float
      Silicate amplitude

    Notes
    -----
    LS18 extinction model. Based on Gao et al. (2013)

    Applicable for Mid-Infrared

    """
    inputs = ('x',)
    outputs = ('y',)#('axav',)

    Tau_si = Parameter(description="Si amplitude", 
        default=0.01, min=0.0, max=10)
    Tau_ice = Parameter(description="Ice amplitude", 
        default=0.4, min=0.0, max=2.5)
    H20_cen_wave = Parameter(description="H2O center wave", 
        default=3.05, bounds=(3.03, 3.07))
    H20_amp = Parameter(description="H2O amplitude", 
        default=1, fixed=True)
    H20_frac_fwhm = Parameter(description="H2O frac fwhm", 
        default=0.312, min=0.2496, max=0.3744)
    # CO2_cen_wave = Parameter(description="CO2 center wave", 
    #     default=4.27, bounds=(4.25, 4.29))
    # CO2_amp = Parameter(description="CO2 amplitude", 
    #     default=0.38, fixed=True)
    # CO2_frac_fwhm = Parameter(description="CO2 frac fwhm", 
    #     default=0.033, min=0.001)
    # CO_cen_wave = Parameter(description="CO center wave", 
    #     default=4.67, bounds=(4.65, 4.69))
    # CO_frac_fwhm = Parameter(description="CO frac fwhm", 
    #     default=0.021, min=0.001)
    # CO_amp = Parameter(description="CO amplitude", 
    #     default=0.55, fixed=True)
    
    @staticmethod
    def kvt(in_x):
        """
        Output the kvt extinction curve
        """
        kvt_wav = np.array([8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.7,
            9.75, 9.8, 10.0, 10.2, 10.4, 10.6,10.8, 11.0, 11.2, 11.4, 
            11.6, 11.8, 12.0, 12.2, 12.4, 12.6, 12.7])

        kvt_int = np.array([.06, .09, .16, .275, .415, .575, .755, .895, .98, .99, 
            1.0, .99, .94, .83, .745, .655, .58, .525, .43, .35, 
            .27, .20, .13, .09, .06, .045, .04314])

        # Extend kvt profile to shorter wavelengths
        kvt_wav_short = in_x[in_x < min(kvt_wav)]
        kvt_int_short_tmp = min(kvt_int) * np.exp(2.03*(kvt_wav_short-min(kvt_wav)))
        # Since kvt_int_shoft_tmp does not reach min(kvt_int), we scale it to stitch it.
        kvt_int_short = kvt_int_short_tmp * (kvt_int[0]/max(kvt_int_short_tmp))

        # Extend kvt profile to longer wavelengths
        kvt_wav_long = in_x[in_x > max(kvt_wav)]
        kvt_int_long = np.zeros(len(kvt_wav_long)) # Need to input the value instead of using zeros

        spline_x = np.concatenate([kvt_wav_short, kvt_wav, kvt_wav_long])
        spline_y = np.concatenate([kvt_int_short, kvt_int, kvt_int_long])

        spline_rep = interpolate.splrep(spline_x, spline_y)
        new_spline_y = interpolate.splev(in_x, spline_rep, der=0)

        ext = drude(in_x, 18, 0.4, 0.247) + new_spline_y

        # Extend to ~2 um
        # assuing beta is 0.1 
        beta = 0.1
        y = (1.0-beta)*ext + beta*(9.7/in_x)**1.7

        return y
    
    # @staticmethod
    # def gao09(in_x):
    #     """
    #     Use attenuation curve from Gao et al. 2009
    #     """
    #     #gao = pd.read_csv('/Users/thomaslai/Documents/astro/PAH/fitter_profile/Gao_2009.csv')
    #     gao = pd.read_table('/Users/thomaslai/Documents/astro/PAH/fitter_profile/Gao09.txt',names=['f','w'])

    #     #gao_tmp = gao[gao.wav > 8]

    #     x = gao.w
    #     y = gao.f
    #     #(gao.f/max(gao_tmp.f)).values # Normalize at 9.7
    #     #spline_rep = interpolate.splrep(spline_x, spline_y)
    #     #new_spline_y = interpolate.splev(in_x, spline_rep, der=0)
    #     f = interpolate.interp1d(x, y)
    #     ynew = f(in_x)
        
    #     return ynew
    
    def evaluate(self, in_x, Tau_si, Tau_ice, 
                 H20_cen_wave, H20_amp, H20_frac_fwhm):#, 
                 #CO2_cen_wave, CO2_amp, CO2_frac_fwhm, 
                 #CO_cen_wave, CO_frac_fwhm, CO_amp):

        # ext = (np.exp(-Tau_si*self.kvt(in_x)) 
        #     * np.exp(-Tau_ice *
        #         (drude(in_x, H20_cen_wave, H20_amp, H20_frac_fwhm) 
        #             + drude(in_x, CO2_cen_wave, CO2_amp, CO2_frac_fwhm)
        #             + drude(in_x, CO_cen_wave, CO_amp, CO_frac_fwhm))))
        
        # ---------
        # Using kvt
        # ---------
        ext = (np.exp(-Tau_si*self.kvt(in_x)) 
            * np.exp(-Tau_ice*(drude(in_x, H20_cen_wave, H20_amp, H20_frac_fwhm))))
                    #+ drude(in_x, CO2_cen_wave, CO2_amp, CO2_frac_fwhm)
                    #+ drude(in_x, CO_cen_wave, CO_amp, CO_frac_fwhm))))
        """
        # -----------
        # Using Gao09
        # -----------
        ext = (np.exp(-Tau_si*self.gao09(in_x)) 
            * np.exp(-Tau_ice *
                (drude(in_x, H20_cen_wave, H20_amp, H20_frac_fwhm))))
        """
        return ext

class bb(Fittable1DModel):

    # Blackbody temperatures
    temp = {
        'temp1': 1500.,
        'temp2': 800.,
        'temp3': 300.,
        'temp4': 200.,
        'temp5': 135.,
        'temp6': 90.,
        'temp7': 65.,
        'temp8': 50.,
        'temp9': 40.,
        'temp10': 35.,
        }

    B1 = Parameter(description="f_T1500",default=5.1950208579398804e-11,min=0.0)
    B2 = Parameter(description="f_T800",default=5.1950208579398804e-10,min=0.0)
    B3 = Parameter(description="f_T300",default=5.1950208579398804e-09,min=0.0)
    B4 = Parameter(description="f_T200",default=9.8026148975804972e-08,min=0.0)
    B5 = Parameter(description="f_T135",default=1.8103439742844785e-06,min=0.0)
    B6 = Parameter(description="f_T90",default=4.1343617340316996e-05,min=0.0)
    B7 = Parameter(description="f_T65",default=0.00032318689045496285,min=0.0)
    B8 = Parameter(description="f_T50",default=0.0029662477318197489,min=0.0)
    B9 = Parameter(description="f_T40",default=0.032730881124734879,min=0.0)
    B10 = Parameter(description="f_T35",default=0.18187057971954346,min=0.0)


    @staticmethod
    def evaluate(in_x, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10):

        y = (B1 * blackbody(in_x, bb.temp['temp1'])
           + B2 * blackbody(in_x, bb.temp['temp2'])
           + B3 * blackbody(in_x, bb.temp['temp3'])
           + B4 * blackbody(in_x, bb.temp['temp4'])
           + B5 * blackbody(in_x, bb.temp['temp5'])
           + B6 * blackbody(in_x, bb.temp['temp6'])
           + B7 * blackbody(in_x, bb.temp['temp7'])
           + B8 * blackbody(in_x, bb.temp['temp8'])
           + B9 * blackbody(in_x, bb.temp['temp9'])
           + B10 * blackbody(in_x, bb.temp['temp9'])
            )

        return y

class sl(Fittable1DModel): 
    """
    Model for starlight calculation
    """
    star_temp = 5000.

    Star_tau = Parameter(description="star_tau_T5000",default=5e-12,min=0.0)


    @staticmethod
    def evaluate(in_x, Star_tau):

        y = Star_tau * blackbody(in_x, sl.star_temp)
    		
        return y

class dust(Fittable1DModel):
	# Dust 
    dust = {
        'D1_cen_wav': 3.29,
        'D1_frac_fwhm': 0.01,
        'D2_cen_wav': 3.4,
        'D2_frac_fwhm': 0.05,
        'D3_cen_wav': 5.27,
        'D3_frac_fwhm': 0.034,
        'D4_cen_wav': 5.7,
        'D4_frac_fwhm': 0.035,
        'D5_cen_wav': 6.22,
        'D5_frac_fwhm': 0.03,
        'D6_cen_wav': 6.69,
        'D6_frac_fwhm': 0.07,
        'D7_cen_wav': 7.42,
        'D7_frac_fwhm': 0.126,
        'D8_cen_wav': 7.6,
        'D8_frac_fwhm': 0.044,
        'D9_cen_wav': 7.85,
        'D9_frac_fwhm': 0.053,
        'D10_cen_wav': 8.33,
        'D10_frac_fwhm': 0.05,
        'D11_cen_wav': 8.61,
        'D11_frac_fwhm': 0.039,
        'D12_cen_wav': 10.68,
        'D12_frac_fwhm': 0.02,
        'D13_cen_wav': 11.23,
        'D13_frac_fwhm': 0.012,
        'D14_cen_wav': 11.33,
        'D14_frac_fwhm': 0.032,
        'D15_cen_wav': 11.99,
        'D15_frac_fwhm': 0.045,
        'D16_cen_wav': 12.62,
        'D16_frac_fwhm': 0.042,
        'D17_cen_wav': 12.69,
        'D17_frac_fwhm': 0.013,
        'D18_cen_wav': 13.48,
        'D18_frac_fwhm': 0.04,
        'D19_cen_wav': 14.04,
        'D19_frac_fwhm': 0.016,
        'D20_cen_wav': 14.19,
        'D20_frac_fwhm': 0.025,
        'D21_cen_wav': 15.9,
        'D21_frac_fwhm': 0.02,
        'D22_cen_wav': 16.45,
        'D22_frac_fwhm': 0.014,
        'D23_cen_wav': 17.04,
        'D23_frac_fwhm': 0.065,
        'D24_cen_wav': 17.37,
        'D24_frac_fwhm': 0.012,
        'D25_cen_wav': 17.87,
        'D25_frac_fwhm': 0.016,
        'D26_cen_wav': 18.92,
        'D26_frac_fwhm': 0.019,
        'D27_cen_wav': 33.1,
        'D27_frac_fwhm': 0.05,
        }
	# cen_wav_list = [D1_cen_wav,  
	# 		   		D2_cen_wav,  
	# 		   		D3_cen_wav,  
	# 		   		D4_cen_wav,  
	# 		   		D5_cen_wav,  
	# 		   		D6_cen_wav,  
	# 		   		D7_cen_wav,  
	# 		   		D8_cen_wav,  
	# 		   		D9_cen_wav,  
	# 		   		D10_cen_wav, 
	# 		   		D11_cen_wav, 
	# 		   		D12_cen_wav, 
	# 		   		D13_cen_wav, 
	# 		   		D14_cen_wav, 
	# 		   		D15_cen_wav, 
	# 		   		D16_cen_wav, 
	# 		   		D17_cen_wav, 
	# 		   		D18_cen_wav, 
	# 		   		D19_cen_wav, 
	# 		   		D20_cen_wav, 
	# 		   		D21_cen_wav, 
	# 		   		D22_cen_wav, 
	# 		   		D23_cen_wav, 
	# 		   		D24_cen_wav, 
	# 		   		D25_cen_wav, 
	# 		   		D26_cen_wav, 
	# 		   		D27_cen_wav
	# 		   		]

	# frac_fwhm_list = [D1_frac_fwhm,
	# 				  D2_frac_fwhm,
	# 				  D3_frac_fwhm,
	# 				  D4_frac_fwhm,
	# 				  D5_frac_fwhm,
	# 				  D6_frac_fwhm,
	# 				  D7_frac_fwhm,
	# 				  D8_frac_fwhm,
	# 				  D9_frac_fwhm,
	# 				  D10_frac_fwhm,
	# 				  D11_frac_fwhm,
	# 				  D12_frac_fwhm,
	# 				  D13_frac_fwhm,
	# 				  D14_frac_fwhm,
	# 				  D15_frac_fwhm,
	# 				  D16_frac_fwhm,
	# 				  D17_frac_fwhm,
	# 				  D18_frac_fwhm,
	# 				  D19_frac_fwhm,
	# 				  D20_frac_fwhm,
	# 				  D21_frac_fwhm,
	# 				  D22_frac_fwhm,
	# 				  D23_frac_fwhm,
	# 				  D24_frac_fwhm,
	# 				  D25_frac_fwhm,
	# 				  D26_frac_fwhm,
	# 				  D27_frac_fwhm
	# 				  ]



	# D_dict = {}
	# for i in range(1, 28):
	# 	D_dict["D{}_cen_wav".format(i)] = D{}_cen_wav.format(i)
	# D_dict = {'D1_cen_wav': D1_cen_wav,
	# 		  'D1_frac_fwhm'

	# 		  }

    D1_int = Parameter(description='D1_int',default=100.,min=0.0)
    D2_int = Parameter(description='D2_int',default=100.,min=0.0)
    D3_int = Parameter(description='D3_int',default=100.,min=0.0)
    D4_int = Parameter(description='D4_int',default=100.,min=0.0)
    D5_int = Parameter(description='D5_int',default=100.,min=0.0)
    D6_int = Parameter(description='D6_int',default=100.,min=0.0)
    D7_int = Parameter(description='D7_int',default=100.,min=0.0)
    D8_int = Parameter(description='D8_int',default=100.,min=0.0)
    D9_int = Parameter(description='D9_int',default=100.,min=0.0)
    D10_int = Parameter(description='D10_int',default=100.,min=0.0)
    D11_int = Parameter(description='D11_int',default=100.,min=0.0)
    D12_int = Parameter(description='D12_int',default=100.,min=0.0)
    D13_int = Parameter(description='D13_int',default=100.,min=0.0)
    D14_int = Parameter(description='D14_int',default=100.,min=0.0)
    D15_int = Parameter(description='D15_int',default=100.,min=0.0)
    D16_int = Parameter(description='D16_int',default=100.,min=0.0)
    D17_int = Parameter(description='D17_int',default=100.,min=0.0)
    D18_int = Parameter(description='D18_int',default=100.,min=0.0)
    D19_int = Parameter(description='D19_int',default=100.,min=0.0)
    D20_int = Parameter(description='D20_int',default=100.,min=0.0)
    D21_int = Parameter(description='D21_int',default=100.,min=0.0)
    D22_int = Parameter(description='D22_int',default=100.,min=0.0)
    D23_int = Parameter(description='D23_int',default=100.,min=0.0)
    D24_int = Parameter(description='D24_int',default=100.,min=0.0)
    D25_int = Parameter(description='D25_int',default=100.,min=0.0)
    D26_int = Parameter(description='D26_int',default=100.,min=0.0)
    D27_int = Parameter(description='D27_int',default=100.,min=0.0)

	# inten_list = [D1_int, 
	# 		    D2_int, 
	# 		    D3_int, 
	# 		    D4_int, 
	# 		    D5_int, 
	# 		    D6_int, 
	# 		    D7_int, 
	# 		    D8_int, 
	# 		    D9_int, 
	# 		    D10_int,
	# 		    D11_int,
	# 		    D12_int,
	# 		    D13_int,
	# 		    D14_int,
	# 		    D15_int,
	# 		    D16_int,
	# 		    D17_int,
	# 		    D18_int,
	# 		    D19_int,
	# 		    D20_int,
	# 		    D21_int,
	# 		    D22_int,
	# 		    D23_int,
	# 		    D24_int,
	# 		    D25_int,
	# 		    D26_int,
	# 		    D27_int
	# 			]

	# def drude(self, in_lam, cen_wave, inten, frac_fwhm):
	# 	return inten*frac_fwhm**2 / ((in_lam/cen_wave-cen_wave/in_lam)**2 + frac_fwhm**2)

    @staticmethod
    def evaluate(in_x,
        D1_int, 
        D2_int, 
        D3_int, 
        D4_int, 
        D5_int, 
        D6_int, 
        D7_int, 
        D8_int, 
        D9_int, 
        D10_int,
        D11_int,
        D12_int,
        D13_int,
        D14_int,
        D15_int,
        D16_int,
        D17_int,
        D18_int,
        D19_int,
        D20_int,
        D21_int,
        D22_int,
        D23_int,
        D24_int,
        D25_int,
        D26_int,
        D27_int
        ):


		# dust = np.zeros(len(in_x))
		# for i in range(1,28):
		# 	cen_wav = cen_wav_list[i]
		# 	inten = inten_list[i]
		# 	frac_fwhm = frac_fwhm_list[i]
		# 	#dust += drude(in_x, cen_wav, inten, frac_fwhm)
		# 	dust += drude(in_x, cen_wave, inten, frac_fwhm)

        y = (drude(in_x, dust.dust['D1_cen_wav'], D1_int, dust.dust['D1_frac_fwhm'])
            + drude(in_x, dust.dust['D2_cen_wav'], D2_int, dust.dust['D2_frac_fwhm'])
            + drude(in_x, dust.dust['D3_cen_wav'], D3_int, dust.dust['D3_frac_fwhm'])
            + drude(in_x, dust.dust['D4_cen_wav'], D4_int, dust.dust['D4_frac_fwhm'])
            + drude(in_x, dust.dust['D5_cen_wav'], D5_int, dust.dust['D5_frac_fwhm'])
            + drude(in_x, dust.dust['D6_cen_wav'], D6_int, dust.dust['D6_frac_fwhm'])
            + drude(in_x, dust.dust['D7_cen_wav'], D7_int, dust.dust['D7_frac_fwhm'])
            + drude(in_x, dust.dust['D8_cen_wav'], D8_int, dust.dust['D8_frac_fwhm'])
            + drude(in_x, dust.dust['D9_cen_wav'], D9_int, dust.dust['D9_frac_fwhm'])
            + drude(in_x, dust.dust['D10_cen_wav'], D10_int, dust.dust['D10_frac_fwhm'])
            + drude(in_x, dust.dust['D11_cen_wav'], D11_int, dust.dust['D11_frac_fwhm'])
            + drude(in_x, dust.dust['D12_cen_wav'], D12_int, dust.dust['D12_frac_fwhm'])
            + drude(in_x, dust.dust['D13_cen_wav'], D13_int, dust.dust['D13_frac_fwhm'])
            + drude(in_x, dust.dust['D14_cen_wav'], D14_int, dust.dust['D14_frac_fwhm'])
            + drude(in_x, dust.dust['D15_cen_wav'], D15_int, dust.dust['D15_frac_fwhm'])
            + drude(in_x, dust.dust['D16_cen_wav'], D16_int, dust.dust['D16_frac_fwhm'])
            + drude(in_x, dust.dust['D17_cen_wav'], D17_int, dust.dust['D17_frac_fwhm'])
            + drude(in_x, dust.dust['D18_cen_wav'], D18_int, dust.dust['D18_frac_fwhm'])
            + drude(in_x, dust.dust['D19_cen_wav'], D19_int, dust.dust['D19_frac_fwhm'])
            + drude(in_x, dust.dust['D20_cen_wav'], D20_int, dust.dust['D20_frac_fwhm'])
            + drude(in_x, dust.dust['D21_cen_wav'], D21_int, dust.dust['D21_frac_fwhm'])
            + drude(in_x, dust.dust['D22_cen_wav'], D22_int, dust.dust['D22_frac_fwhm'])
            + drude(in_x, dust.dust['D23_cen_wav'], D23_int, dust.dust['D23_frac_fwhm'])
            + drude(in_x, dust.dust['D24_cen_wav'], D24_int, dust.dust['D24_frac_fwhm'])
            + drude(in_x, dust.dust['D25_cen_wav'], D25_int, dust.dust['D25_frac_fwhm'])
            + drude(in_x, dust.dust['D26_cen_wav'], D26_int, dust.dust['D26_frac_fwhm'])
            + drude(in_x, dust.dust['D27_cen_wav'], D27_int, dust.dust['D27_frac_fwhm'])
            )

        return y


class line(Fittable1DModel):
	# Line
    line = {
        'L1_cen_wav': 2.63,
        'L1_fwhm': 0.01,
        'L2_cen_wav': 3.75,
        'L2_fwhm': 0.01,
        'L3_cen_wav': 4.05,
        'L3_fwhm': 0.01,
        'L4_cen_wav': 4.65,
        'L4_fwhm': 0.01,
        'L5_cen_wav': 5.5115,
        'L5_fwhm': 0.053,
        'L6_cen_wav': 6.1088,
        'L6_fwhm': 0.053,
        'L7_cen_wav': 6.985274,
        'L7_fwhm': 0.01,
        'L8_cen_wav': 8.0258,
        'L8_fwhm': 0.01,
        'L9_cen_wav': 8.99138,
        'L9_fwhm': 0.01,
        'L10_cen_wav': 9.6649,
        'L10_fwhm': 0.01,
        'L11_cen_wav': 10.5105,
        'L11_fwhm': 0.01,
        'L12_cen_wav': 12.2785,
        'L12_fwhm': 0.01,
        'L13_cen_wav': 12.813,
        'L13_fwhm': 0.01,
        'L14_cen_wav': 15.555,
        'L14_fwhm': 0.014,
        'L15_cen_wav': 17.0346,
        'L15_fwhm': 0.014,
        'L16_cen_wav': 18.713,
        'L16_fwhm': 0.014,
        'L17_cen_wav': 25.91,
        'L17_fwhm': 0.013,
        'L18_cen_wav': 25.989,
        'L18_fwhm': 0.01,
        'L19_cen_wav': 28.2207,
        'L19_fwhm': 0.01,
        'L20_cen_wav': 33.480,
        'L20_fwhm': 0.01,
        'L21_cen_wav': 34.8152,
        'L21_fwhm': 0.01,
        }

	# cen_wav_list = [L1_cen_wav,  
	# 		   		L2_cen_wav,  
	# 		   		L3_cen_wav,  
	# 		   		L4_cen_wav,  
	# 		   		L5_cen_wav,  
	# 		   		L6_cen_wav,  
	# 		   		L7_cen_wav,  
	# 		   		L8_cen_wav,  
	# 		   		L9_cen_wav,  
	# 		   		L10_cen_wav, 
	# 		   		L11_cen_wav, 
	# 		   		L12_cen_wav, 
	# 		   		L13_cen_wav, 
	# 		   		L14_cen_wav, 
	# 		   		L15_cen_wav, 
	# 		   		L16_cen_wav, 
	# 		   		L17_cen_wav, 
	# 		   		L18_cen_wav, 
	# 		   		L19_cen_wav, 
	# 		   		L20_cen_wav, 
	# 		   		L21_cen_wav, 
	# 		   		]

	# frac_fwhm_list = [L1_frac_fwhm,
	# 				  L2_frac_fwhm,
	# 				  L3_frac_fwhm,
	# 				  L4_frac_fwhm,
	# 				  L5_frac_fwhm,
	# 				  L6_frac_fwhm,
	# 				  L7_frac_fwhm,
	# 				  L8_frac_fwhm,
	# 				  L9_frac_fwhm,
	# 				  L10_frac_fwhm,
	# 				  L11_frac_fwhm,
	# 				  L12_frac_fwhm,
	# 				  L13_frac_fwhm,
	# 				  L14_frac_fwhm,
	# 				  L15_frac_fwhm,
	# 				  L16_frac_fwhm,
	# 				  L17_frac_fwhm,
	# 				  L18_frac_fwhm,
	# 				  L19_frac_fwhm,
	# 				  L20_frac_fwhm,
	# 				  L21_frac_fwhm,
	# 				  ]



	# D_dict = {}
	# for i in range(1, 28):
	# 	D_dict["D{}_cen_wav".format(i)] = D{}_cen_wav.format(i)
	# D_dict = {'D1_cen_wav': D1_cen_wav,
	# 		  'D1_frac_fwhm'

	# 		  }

    L1_int = Parameter(description='L1_int',default=10.,min=0.0)# old default = 0.001
    L2_int = Parameter(description='L2_int',default=10.,min=0.0)# old default = 0.001
    L3_int = Parameter(description='L3_int',default=10.,min=0.0)# old default = 0.001
    L4_int = Parameter(description='L4_int',default=10.,min=0.0)# old default = 0.001
    L5_int = Parameter(description='L5_int',default=100,min=0.0) # old default = 0.01
    L6_int = Parameter(description='L6_int',default=100,min=0.0) # old default = 0.01  
    L7_int = Parameter(description='L7_int',default=100,min=0.0) # old default = 0.01
    L8_int = Parameter(description='L8_int',default=100,min=0.0) # old default = 0.01
    L9_int = Parameter(description='L9_int',default=100,min=0.0) # old default = 0.01
    L10_int = Parameter(description='L10_int',default=100,min=0.0)
    L11_int = Parameter(description='L11_int',default=100,min=0.0)
    L12_int = Parameter(description='L12_int',default=100,min=0.0)
    L13_int = Parameter(description='L13_int',default=100,min=0.0)
    L14_int = Parameter(description='L14_int',default=100,min=0.0)
    L15_int = Parameter(description='L15_int',default=100,min=0.0)
    L16_int = Parameter(description='L16_int',default=100,min=0.0)
    L17_int = Parameter(description='L17_int',default=100,min=0.0)
    L18_int = Parameter(description='L18_int',default=100,min=0.0)
    L19_int = Parameter(description='L19_int',default=100,min=0.0)
    L20_int = Parameter(description='L20_int',default=100,min=0.0)
    L21_int = Parameter(description='L21_int',default=100,min=0.0)

	# inten_list = [L1_int, 
	# 		    L2_int, 
	# 		    L3_int, 
	# 		    L4_int, 
	# 		    L5_int, 
	# 		    L6_int, 
	# 		    L7_int, 
	# 		    L8_int, 
	# 		    L9_int, 
	# 		    L10_int,
	# 		    L11_int,
	# 		    L12_int,
	# 		    L13_int,
	# 		    L14_int,
	# 		    L15_int,
	# 		    L16_int,
	# 		    L17_int,
	# 		    L18_int,
	# 		    L19_int,
	# 		    L20_int,
	# 		    L21_int,
	# 			]



    @staticmethod
    def evaluate(in_x,
                L1_int, 
                L2_int, 
                L3_int, 
                L4_int, 
                L5_int, 
                L6_int, 
                L7_int, 
                L8_int, 
                L9_int, 
                L10_int,
                L11_int,
                L12_int,
                L13_int,
                L14_int,
                L15_int,
                L16_int,
                L17_int,
                L18_int,
                L19_int,
                L20_int,
                L21_int,
                ):


		# dust = np.zeros(len(in_x))
		# for i in range(1,28):
		# 	cen_wav = cen_wav_list[i]
		# 	inten = inten_list[i]
		# 	frac_fwhm = frac_fwhm_list[i]
		# 	#dust += drude(in_x, cen_wav, inten, frac_fwhm)
		# 	dust += drude(in_x, cen_wave, inten, frac_fwhm)

		# Line
        y = (gaussian(in_x, line.line['L1_cen_wav'], L1_int, line.line['L1_fwhm'])
            + gaussian(in_x, line.line['L2_cen_wav'], L2_int, line.line['L2_fwhm'])
            + gaussian(in_x, line.line['L3_cen_wav'], L3_int, line.line['L3_fwhm'])
            + gaussian(in_x, line.line['L4_cen_wav'], L4_int, line.line['L4_fwhm'])
            + gaussian(in_x, line.line['L5_cen_wav'], L5_int, line.line['L5_fwhm'])
            + gaussian(in_x, line.line['L6_cen_wav'], L6_int, line.line['L6_fwhm'])
            + gaussian(in_x, line.line['L7_cen_wav'], L7_int, line.line['L7_fwhm'])
            + gaussian(in_x, line.line['L8_cen_wav'], L8_int, line.line['L8_fwhm'])
            + gaussian(in_x, line.line['L9_cen_wav'], L9_int, line.line['L9_fwhm'])
            + gaussian(in_x, line.line['L10_cen_wav'], L10_int, line.line['L10_fwhm'])
            + gaussian(in_x, line.line['L11_cen_wav'], L11_int, line.line['L11_fwhm'])
            + gaussian(in_x, line.line['L12_cen_wav'], L12_int, line.line['L12_fwhm'])
            + gaussian(in_x, line.line['L13_cen_wav'], L13_int, line.line['L13_fwhm'])
            + gaussian(in_x, line.line['L14_cen_wav'], L14_int, line.line['L14_fwhm'])
            + gaussian(in_x, line.line['L15_cen_wav'], L15_int, line.line['L15_fwhm'])
            + gaussian(in_x, line.line['L16_cen_wav'], L16_int, line.line['L16_fwhm'])
            + gaussian(in_x, line.line['L17_cen_wav'], L17_int, line.line['L17_fwhm'])
            + gaussian(in_x, line.line['L18_cen_wav'], L18_int, line.line['L18_fwhm'])
            + gaussian(in_x, line.line['L19_cen_wav'], L19_int, line.line['L19_fwhm'])
            + gaussian(in_x, line.line['L20_cen_wav'], L20_int, line.line['L20_fwhm'])
            + gaussian(in_x, line.line['L21_cen_wav'], L21_int, line.line['L21_fwhm'])
            )

        return y


def composite_conti(in_x, fit_result):
    key = fit_result.param_names
    bb_key_idx = [i for i, k in enumerate(key) if k[0]=='B']

    y = np.zeros(len(in_x))
    for B_idx, all_idx in enumerate(bb_key_idx):
        B_idx += 1 # bb idx starts from 1
        B = fit_result.parameters[all_idx]
        y += B * blackbody(in_x, bb.temp["temp{}".format(B_idx)])

    sl_key_idx = [i for i, k in enumerate(key) if k[0]=='S']
    Star_tau = result.parameters[sl_key_idx]
    
    y += Star_tau * blackbody(in_x, 5000)

    return y

def composite_dust(in_x, fit_result):
    key = fit_result.param_names
    dust_key_idx = [i for i, k in enumerate(key) if k[0]=='D']

    y = np.zeros(len(in_x))
    for D_idx, all_idx in enumerate(dust_key_idx):
        D_idx += 1 # dust idx start from 1
        dust_int = fit_result.parameters[all_idx]
        y += drude(x, dust.dust["D{}_cen_wav".format(D_idx)], dust_int, dust.dust["D{}_frac_fwhm".format(D_idx)])
       
    return y


def composite_att_S07(in_x, fit_result):
    """
    Output the attenuation curve normalized at 1
    """
    key = fit_result.param_names
    att_key_idx = [i for i, k in enumerate(key) if k[0:6]=='Tau_si']

    y = np.exp(-fit_result.parameters[att_key_idx]*S07.kvt(in_x))
    
    return y



def composite_att_LS18(in_x, fit_result):
    """
    Output the attenuation curve normalized at 1
    """
    key = fit_result.param_names
    att_key_idx = [i for i, k in enumerate(key) if k[0]=='T']

    #H2O_cen_wave_idx = key.index('H2O_cen_wave_0')

    #H20_cen_wave = fit_result.parameters[H2O_key_idx]
    #att_key_idx[0] = 'Tau_si_0'
    #att_key_idx[1] = 'Tau_ice_0'
    
    # ---------
    # Using kvt
    # ---------
    y = (np.exp(-fit_result.parameters[att_key_idx[0]]*LS18.kvt(in_x)) * 
        np.exp(-fit_result.parameters[att_key_idx[1]]*(drude(in_x, LS18().H20_cen_wave, LS18().H20_amp, LS18().H20_frac_fwhm))))
    
    # -----------
    # Using Gao09
    # -----------
    #y = (np.exp(-fit_result.parameters[att_key_idx[0]]*LS18.gao09(in_x)) * 
    #    np.exp(-fit_result.parameters[att_key_idx[1]]*(drude(in_x, LS18().H20_cen_wave, LS18().H20_amp, LS18().H20_frac_fwhm))))
    
    return y

# Example 1:
# df = pd.read_table('/Users/thomaslai/Documents/astro/PAH/JWST_GO1/sample_spec/comb_norm_spec_icelt0.15.txt',header=None,names=['w','f','f_l','f_h'])
# x = df.w.values
# y = df.f.values

# Example 2
spec_dir = './data/'
df = pd.read_table(spec_dir+'1120229'+'.txt',header=None,names=['w','f','f_l','f_h'])
x = df.w.values
y = df.f.values


# guess star tau
spline_rep = interpolate.splrep(x, y)
y_spec = interpolate.splev(5.5, spline_rep, der=0)

spline_rep = interpolate.splrep(x, blackbody(x,5000))
y_bb = interpolate.splev(5.5, spline_rep, der=0)

guess_tau_star = y_spec / y_bb 
guess_tau_star = 6e-12
# Model build up
init_S07 = S07() * (bb()+sl(guess_tau_star)+dust()+line())

#init_LS18 = (bb()+sl(guess_tau_star)+dust()+line())
init_LS18 = LS18()*(bb()+sl(guess_tau_star)+dust()+line())
# ==================
# Running the fitter
# ==================
init = init_S07
#init = init_LS18
fit = LevMarLSQFitter()
result = fit(init, x, y, maxiter=5000)
# ==================

plt.close(1)
fig = plt.figure(1)
ax = fig.add_subplot(111)
# Plot data points
ax.scatter(x, y/x, s=10,marker='s',facecolors='none', edgecolors='k')
ax.plot(x, result(x)/x, 'g')



key = result.param_names

# Composite attenuation curve: S07 or LS18
if init == init_S07:
    comp_att = composite_att_S07
if init == init_LS18:
    comp_att = composite_att_LS18

# -----------------------
# Plot blackbody elements
# -----------------------
bb_key_idx = [i for i, k in enumerate(key) if k[0]=='B']

for B_idx, all_idx in enumerate(bb_key_idx):
   B_idx += 1 # bb idx start from 1
   B = result.parameters[all_idx]
   plt.plot(x ,B * blackbody(x, bb.temp["temp{}".format(B_idx)]) * comp_att(x, result)/x,color='red')

# -----------------------
# Plot starlight elements
# -----------------------
sl_key_idx = [i for i, k in enumerate(key) if k[0]=='S']
Star_tau = result.parameters[sl_key_idx]
plt.plot(x ,Star_tau * blackbody(x, 5000) * comp_att(x, result)/x, color='cyan', zorder=0)

# -----------------------
# Plot continuum (bb+sl)
# -----------------------
plt.plot(x, composite_conti(x, result) * comp_att(x, result)/x, color='gray', linewidth=1.5, zorder=3)

# -----------------------
# Plot dust features
# -----------------------
dust_key_idx = [i for i, k in enumerate(key) if k[0]=='D']

dust_key_idx = [i for i, k in enumerate(key) if k[0]=='D']

for D_idx, all_idx in enumerate(dust_key_idx):
   D_idx += 1 # dust idx starts from 1
   dust_int = result.parameters[all_idx]
   dust_profile = drude(x, dust.dust["D{}_cen_wav".format(D_idx)], dust_int, dust.dust["D{}_frac_fwhm".format(D_idx)])
   
   plt.plot(x ,(dust_profile+composite_conti(x, result)) * comp_att(x, result)/x, color='purple', linewidth=0.3, zorder=1)

# -----------------------
# Plot line features
# -----------------------
line_key_idx = [i for i, k in enumerate(key) if k[0]=='L']

for L_idx, all_idx in enumerate(line_key_idx):
   L_idx += 1 # line idx starts from 1
   line_int = result.parameters[all_idx]
   line_profile = gaussian(x, line.line["L{}_cen_wav".format(L_idx)], line_int, line.line["L{}_fwhm".format(L_idx)])
   
   plt.plot(x ,(line_profile+composite_conti(x, result)) * comp_att(x, result)/x, color='blue', linewidth=0.3, zorder=1)

# -----------------------
# Plot attenuation curve
# -----------------------
ax2 = ax.twinx()

if init == init_S07:
    ax2.plot(x, composite_att_S07(x, result), linestyle='--', color='k')
if init == init_LS18:
    ax2.plot(x, composite_att_LS18(x, result), linestyle='--', color='k')

ax2.set_ylim([0,1.1])
ax.set_xlim([2.4,30])

y_plot = y/x
ax.set_ylim([-np.max(y_plot[x < 20])*0.2,np.max(y_plot[x < 20])*1.6])

ax.set_xscale('log')
ax.set_ylabel(r'Flux (I$\nu$ / $\lambda$)')
ax.set_xlabel(r'Rest wavelength ($\mu$m)')

ax.set_xticks([3.,4.,5.,10, 15, 20])
ax.xaxis.set_major_formatter(ScalarFormatter())

ax2.set_ylabel(r'extinction ($\tau$)')

plt.show()