from dared.dados_red import DADOSobservation, AlpyArNe
from astropy.io import fits

obs = DADOSobservation(
    flat=fits.getdata(r"testspec/flats/spectrum-0001_flat_afterwards_20s.fit"),
    calib=fits.getdata(r"testspec/wavelength_calib.fit"),
    calib_lamp=AlpyArNe
)
obs.debug_mode = True

obs.findorder(output="orderfind.pdf")
obs.calibrate_wavelengths()