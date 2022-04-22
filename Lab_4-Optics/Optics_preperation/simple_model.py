import numpy as np
import math


muabo = np.genfromtxt("Lab_4-Optics/Optics_preperation/muabo.txt", delimiter=",")
muabd = np.genfromtxt("Lab_4-Optics/Optics_preperation/muabd.txt", delimiter=",")

red_wavelength = 600 # Replace with wavelength in nanometres
green_wavelength = 515 # Replace with wavelength in nanometres
blue_wavelength = 460 # Replace with wavelength in nanometres

red_wavelength_reflectance = 0.5
green_wavelength_reflectance = 0.45
blue_wavelength_reflectance = 0.4

finger_thickness_Carl   = 0.009 # [m]
finger_thickness_Daniel = 0.008 # [m]

wavelength = np.array([red_wavelength, green_wavelength, blue_wavelength])

def mua_blood_oxy(x): return np.interp(x, muabo[:, 0], muabo[:, 1])
def mua_blood_deoxy(x): return np.interp(x, muabd[:, 0], muabd[:, 1])

def constants(wavelength, bvf = 0.01, mua_other = 25):
    #bvf = 0.01 # Blood volume fraction, average blood amount in tissue
    oxy = 0.8 # Blood oxygenations

    # Absorption coefficient ($\mu_a$ in lab text)
    # Units: 1/m
    #mua_other = 25 # Background absorption due to collagen, et cetera
    mua_blood = (mua_blood_oxy(wavelength)*oxy # Absorption due to
                + mua_blood_deoxy(wavelength)*(1-oxy)) # pure blood
    mua = mua_blood*bvf + mua_other

    # reduced scattering coefficient ($\mu_s^\prime$ in lab text)
    # the numerical constants are thanks to N. Bashkatov, E. A. Genina and
    # V. V. Tuchin. Optical properties of skin, subcutaneous and muscle
    # tissues: A review. In: J. Innov. Opt. Health Sci., 4(1):9-38, 2011.
    # Units: 1/m
    musr = 100 * (17.6*(wavelength/500)**-4 + 18.78*(wavelength/500)**-0.22)
    return mua, musr

def pen_depth(wavelength, bvf = 0.01, mua_other = 25):
    mua, musr = constants(wavelength, bvf, mua_other)

    delta = math.sqrt( 1 / (3 * (musr + mua) * mua) )
    return delta

def light_flux_at_depth(depth, wavelength, bvf = 0.01, mua_other = 25):
    delta = pen_depth(wavelength, bvf, mua_other)

    flux = math.exp(-depth/delta)
    return flux

# def reflectance(wavelength):
#     mua, musr = constants(wavelength)
#     reflectance = math.sqrt(3*((musr/mua + 1))) # 1/(pen_depth(wavelength) *mua)
#     return reflectance

def probed_depth(wavelength, reflectance):
    delta = pen_depth(wavelength)

    probed_depth = - math.log(reflectance) * delta / 2 
    return probed_depth

def contrast(high_val, normal_val):
    contrast = abs(high_val - normal_val) / normal_val
    return contrast

def contrast_vein(wavelength):
    c = contrast(light_flux_at_depth(3e-4, wavelength, 1.0, 0.0), light_flux_at_depth(3e-4, wavelength, 0.01))
    return c



print(f"Penetration depth for red waves:\t{pen_depth(red_wavelength) * 1000:.3f}mm")
print(f"Penetration depth for green waves:\t{pen_depth(green_wavelength) * 1000:.3f}mm")
print(f"Penetration depth for blue waves:\t{pen_depth(blue_wavelength) * 1000:.3f}mm")
print("\n\n")

print(f"Flux remaining at other side of Carl's finger, red light:\t{light_flux_at_depth(finger_thickness_Carl, red_wavelength) * 1000:.5f}‰")
print(f"Flux remaining at other side of Daniel's finger, red light:\t{light_flux_at_depth(finger_thickness_Daniel, red_wavelength) * 1000:.5f}‰")
print()
print(f"Flux remaining at other side of Carl's finger, green light:\t{light_flux_at_depth(finger_thickness_Carl, green_wavelength) * 1000:.5f}‰")
print(f"Flux remaining at other side of Daniel's finger, green light:\t{light_flux_at_depth(finger_thickness_Daniel, green_wavelength) * 1000:.5f}‰")
print()
print(f"Flux remaining at other side of Carl's finger, blue light:\t{light_flux_at_depth(finger_thickness_Carl, blue_wavelength) * 1000:.5f}‰")
print(f"Flux remaining at other side of Daniel's finger, blue light:\t{light_flux_at_depth(finger_thickness_Daniel, blue_wavelength) * 1000:.5f}‰")
print("\n\n")

print(f"Probed depth for red waves:\t{probed_depth(red_wavelength, red_wavelength_reflectance) * 1000:.3f}mm")
print(f"Probed depth for green waves:\t{probed_depth(green_wavelength, green_wavelength_reflectance) * 1000:.3f}mm")
print(f"Probed depth for blue waves:\t{probed_depth(blue_wavelength, blue_wavelength_reflectance) * 1000:.3f}mm")
print("\n\n")

print(f"Flux remaining at other side of 300um vein with 100% blood, red light:\t\t{light_flux_at_depth(3e-4, red_wavelength, 1.0, 0.0) * 1000:.3f}‰")
print(f"Flux remaining at other side of 300um vein with 100% blood, green light:\t{light_flux_at_depth(3e-4, green_wavelength, 1.0, 0.0) * 1000:.3f}‰")
print(f"Flux remaining at other side of 300um vein with 100% blood, blue light:\t\t{light_flux_at_depth(3e-4, blue_wavelength, 1.0, 0.0) * 1000:.3f}‰")
print()
print(f"Flux remaining at other side of 300um vein with 1% blood, red light:\t\t{light_flux_at_depth(3e-4, red_wavelength, 0.01) * 1000:.3f}‰")
print(f"Flux remaining at other side of 300um vein with 1% blood, green light:\t\t{light_flux_at_depth(3e-4, green_wavelength, 0.01) * 1000:.3f}‰")
print(f"Flux remaining at other side of 300um vein with 1% blood, blue light:\t\t{light_flux_at_depth(3e-4, blue_wavelength, 0.01) * 1000:.3f}‰")
print()
print(f"Contrast with 300um vein, 100% blood and 1% blood, red light:\t\t\t{contrast_vein(red_wavelength):.5f}")
print(f"Contrast with 300um vein, 100% blood and 1% blood, green light:\t\t\t{contrast_vein(green_wavelength):.5f}")
print(f"Contrast with 300um vein, 100% blood and 1% blood, blue light:\t\t\t{contrast_vein(blue_wavelength):.5f}")
print("\n\n")

# print(light_flux_at_depth(3e-4, red_wavelength, 1.0, 0.0))
# print(light_flux_at_depth(3e-4, red_wavelength, 0.01))
