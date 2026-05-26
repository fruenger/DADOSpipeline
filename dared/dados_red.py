import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from photutils.aperture import RectangularAperture, aperture_photometry
from tqdm import tqdm
from itertools import combinations
from scipy.signal import find_peaks, peak_widths
from scipy.special import comb
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d
from pathlib import Path
from PIL import Image

from .utils import Formatter, get_timestamp, wavelength_to_rgb

fmt = Formatter()


class Spectrograph:
    """
    Hardware description of a spectrograph instrument.

    Stores the physical characteristics of a spectrograph needed for
    order identification and wavelength calibration.

    Parameters
    ----------
    order_width : float
        The spatial width of one spectral order on the detector, in microns.
    Norders : int
        The number of spectral orders produced by the spectrograph.
    wavelength_dispersion : float
        The wavelength dispersion of the spectrograph in Angstrom per micron.

    Attributes
    ----------
    order_width : float
    Norders : int
    wavelength_dispersion : float

    Examples
    --------
    >>> my_spec = Spectrograph(order_width=1128., Norders=3, wavelength_dispersion=0.104)
    >>> my_spec.order_width
    1128.0
    """
    def __init__(self, order_width, Norders, wavelength_dispersion):

        self.order_width = order_width
        self.Norders     = Norders
        self.wavelength_dispersion = wavelength_dispersion

DADOS = Spectrograph(
    order_width=1128., # micron
    Norders=3,
    wavelength_dispersion=0.104 # Angstrom/micron
)

class Camera:
    """
    Hardware description of a CCD/CMOS camera detector.

    Stores the physical and quantum efficiency characteristics of a camera,
    including a callable interpolation of its response (QE) curve.

    Parameters
    ----------
    response_curve : np.ndarray, shape (n, 2)
        A two-column array where the first column contains wavelengths in
        Angstrom and the second column contains the corresponding quantum
        efficiency values (0-1). Used to build a 1D interpolation function.
    pixel_size : float
        Physical size of one pixel in microns.
    chip_model : str, optional
        A human-readable identifier for the detector chip (e.g. ``"IMX571"``).
        Defaults to ``""``.

    Attributes
    ----------
    response_curve : scipy.interpolate.interp1d
        Callable interpolant of the QE curve. Returns 0 outside the
        provided wavelength bounds.
    pixel_size : float
        Pixel size in microns.
    chip_model : str

    Examples
    --------
    >>> import numpy as np
    >>> qe_data = np.array([[4000., 0.7], [5000., 0.85], [6000., 0.75]])
    >>> cam = Camera(response_curve=qe_data, pixel_size=3.76, chip_model="TestChip")
    >>> cam.response_curve(5000.)
    array(0.85)
    >>> cam.pixel_size
    3.76
    """
    def __init__(self, response_curve, pixel_size, chip_model=""):
        """Response curve is a (n, 2) numpy array that defines an interpolation grid of the camera response curve"""
        self.response_curve = interp1d(*response_curve.T, bounds_error=False, fill_value=0)
        self.chip_model = chip_model
        self.pixel_size = pixel_size # microns



ST8 = Camera(
    response_curve = np.array([
        [3500.00,          0.205447470817120],
        [3601.4492753623,  0.154085603112840],
        [3698.2872200263,  0.224124513618677],
        [3804.3478260869,  0.242023346303502],
        [3910.4084321475,  0.284824902723735],
        [4007.2463768115,  0.284046692607003],
        [4113.3069828722,  0.297276264591439],
        [4196.3109354413,  0.299610894941634],
        [4293.1488801054,  0.296498054474708],
        [4496.0474308300,  0.370428015564202],
        [4597.4967061923,  0.390661478599221],
        [4698.9459815546,  0.392996108949416],
        [4805.0065876152,  0.383657587548638],
        [4901.8445322793,  0.382101167315175],
        [5003.2938076416,  0.384435797665369],
        [5104.7430830039,  0.400778210116731],
        [5206.1923583662,  0.437354085603112],
        [5390.6455862977,  0.541634241245136],
        [5593.5441370223,  0.569649805447470],
        [5694.9934123847,  0.598443579766537],
        [5796.4426877470,  0.615564202334630],
        [5897.8919631093,  0.618677042801556],
        [6003.9525691699,  0.614785992217898],
        [6202.2397891963,  0.592996108949416],
        [6303.6890645586,  0.573540856031128],
        [6405.1383399209,  0.550194552529182],
        [6511.1989459815,  0.544747081712062],
        [6612.6482213438,  0.536186770428015],
        [6695.6521739130,  0.536186770428015],
        [6806.3241106719,  0.513618677042801],
        [6898.5507246376,  0.498832684824902],
        [6990.7773386034,  0.480155642023346],
        [7101.4492753623,  0.456031128404669],
        [7212.1212121212,  0.439688715953307],
        [7410.4084321475,  0.417898832684825],
        [7507.2463768115,  0.416342412451361],
        [7608.6956521739,  0.417898832684825],
        [7714.7562582345,  0.415564202334630],
        [7811.5942028985,  0.400778210116731],
        [8000.6587615283,  0.347081712062256],
        [8092.8853754940,  0.328404669260700],
        [8198.9459815546,  0.319844357976653],
        [8295.7839262187,  0.307392996108949],
        [8401.8445322793,  0.307392996108949],
        [8498.6824769433,  0.298054474708171],
        [8609.3544137022,  0.293385214007782],
        [8701.5810276679,  0.283268482490272],
        [8812.2529644268,  0.263813229571984],
        [9005.9288537549,  0.225680933852140],
        [9107.3781291172,  0.208560311284046],
        [9199.6047430830,  0.181322957198443],
        [9812.9117259552,  0.078599221789883],
        [10011.198945981,  0.050583657587548],
        [10117.259552042,  0.038910505836575],
        [10306.324110671,  0.022568093385214],
        [10407.773386034,  0.016342412451361],
        [10689.064558629,  0.008560311284046],
        [10804.347826086,  0.007782101167315],
        [10891.963109354,  0.006225680933852],
        [11002.635046113,  0.005447470817120]
    ]),
    pixel_size=9., # microns
    chip_model = "KAF-1602E"
)


QHY268M = Camera(
    response_curve=np.array([
        [3999.3064523843,  0.712871287128712],
        [4088.8154032794,  0.769576957695769],
        [4132.3523656713,  0.793879387938794],
        [4149.2714488840,  0.798379837983798],
        [4192.7953664931,  0.817281728172817],
        [4214.5714571457,  0.832583258325832],
        [4231.5057592715,  0.843384338433843],
        [4296.7775038373,  0.865886588658865],
        [4395.8700217847,  0.890189018901890],
        [4502.2045682829,  0.912691269126912],
        [4540.8671301912,  0.918991899189918],
        [4622.9905599255,  0.918091809180918],
        [4659.2159215921,  0.915391539153915],
        [4702.6789635485,  0.909090909090909],
        [4724.4028750701,  0.902790279027902],
        [4784.7806519782,  0.899189918991899],
        [4837.9120520747,  0.895589558955895],
        [4862.0601190553,  0.892889288928892],
        [4881.3729199006,  0.888388838883888],
        [4891.0282332581,  0.885688568856885],
        [4920.0202628958,  0.888388838883888],
        [4987.6748544419,  0.897389738973897],
        [5004.5852411328,  0.898289828982898],
        [5052.8857233549,  0.894689468946894],
        [5130.1673645625,  0.889288928892889],
        [5166.3818555768,  0.882088208820881],
        [5190.5299225574,  0.879387938793879],
        [5236.4105975814,  0.873987398739874],
        [5316.0903046826,  0.861386138613861],
        [5393.3480304552,  0.846084608460846],
        [5506.8245955030,  0.825382538253825],
        [5673.4173417341,  0.794779477947794],
        [5808.6134700426,  0.765976597659765],
        [5953.4431704039,  0.725472547254725],
        [6112.7569278666,  0.681368136813681],
        [6286.5873543876,  0.647164716471647],
        [6402.4554629375,  0.616561656165616],
        [6460.3851689516,  0.599459945994599],
        [6530.3812989994,  0.577857785778577],
        [6634.1873317766,  0.553555355535553],
        [6701.7810476699,  0.537353735373537],
        [6728.3467477182,  0.535553555355535],
        [6764.5612387325,  0.528352835283528],
        [6779.0387734425,  0.522052205220522],
        [6827.3262108819,  0.513051305130513],
        [6882.8513286111,  0.500450045004500],
        [6935.9479426203,  0.482448244824482],
        [6984.2136387551,  0.464446444644464],
        [7056.6360983924,  0.447344734473447],
        [7182.1703909521,  0.418541854185418],
        [7295.6621749131,  0.404140414041403],
        [7404.3012996951,  0.380738073807380],
        [7455.0085443326,  0.373537353735373],
        [7532.2640959748,  0.357335733573357],
        [7648.1778612643,  0.345634563456345],
        [7751.9860681720,  0.322232223222322],
        [7848.5631171812,  0.305130513051305],
        [7879.9619092344,  0.304230423042304],
        [8019.9889554172,  0.275427542754275],
        [8044.1435447892,  0.275427542754275],
        [8085.2063467216,  0.275427542754275],
        [8104.5278440887,  0.274527452745274],
        [8193.8650386777,  0.260126012601259],
        [8237.3150358514,  0.248424842484248],
        [8278.3691412619,  0.244824482448244],
        [8336.3336333633,  0.242124212421242],
        [8379.8118942329,  0.242124212421242],
        [8585.0041525891,  0.191719171917191],
        [8717.8456976132,  0.188118811881188],
        [8754.0601886275,  0.180918091809180],
        [8826.4783000039,  0.162016201620161],
        [8877.1703257282,  0.148514851485148],
        [8966.5401322740,  0.147614761476147],
        [9000.3630797862,  0.150315031503150],
        [9048.6700843997,  0.149414941494149],
        [9087.3043826121,  0.144014401440143],
        [9142.8316744717,  0.132313231323132],
        [9251.4729733842,  0.109810981098109],
        [9386.7343256064,  0.108010801080108],
        [9478.4956756545,  0.097209720972097],
        [9524.3676541567,  0.088208820882088],
        [9560.5734486492,  0.077407740774077],
        [9613.6961522239,  0.070207020702070],
        [9681.3181318131,  0.065706570657065],
        [9775.5145079725,  0.063006300630063],
        [9862.4623331898,  0.059405940594059],
        [9927.6666797114,  0.054005400540053],
        [9983.2070163538,  0.047704770477047]
    ]),
    pixel_size=3.76, # microns
    chip_model="IMX571"
)


def findorder(flat_frame, owidth, Norders, output=None, debug=False, order_detection_threshold=None, order_poly_fit_degree=3, maxiter=10):
    """
    Locate spectral orders in a flat-field image.

    Convolves the flat frame with a top-hat kernel of width `owidth` along
    the spatial (row) axis, then identifies the `Norders` brightest peaks in
    each pixel column. The peak centroids are fitted with polynomials to
    describe the curved trajectory of each order across the detector.

    Parameters
    ----------
    flat_frame : np.ndarray, shape (ny, nx)
        A 2-D flat-field image array.
    owidth : float
        Expected order width in pixels (typically ``spectrograph.order_width /
        camera.pixel_size``).
    Norders : int
        Number of spectral orders to locate.
    output : str, optional
        File path for the diagnostic figure. Pass ``""`` (default) to skip
        saving.
    debug : bool, optional
        If ``True``, renders and optionally saves a diagnostic figure showing
        the flat image overlaid with the detected order centres and boundaries.
        Defaults to ``False``.
    order_detection_threshold : float or None, optional
        Minimum peak height for order detection. Currently unused in the
        implementation; reserved for future use. Defaults to ``None``.
    order_poly_fit_degree : int, optional
        Degree of the polynomial used to fit each order's centre as a function
        of pixel column. Defaults to ``3``.
    maxiter : int, optional
        Maximum number of fitting iterations. Currently unused; reserved for
        future use. Defaults to ``10``.

    Returns
    -------
    order_center_fits : list of np.poly1d, length Norders
        One callable polynomial per order. Each polynomial maps a pixel column
        index ``x`` to the estimated row centre ``y`` of that order,
        i.e. ``order_center_fits[i](x)`` returns the row position of order
        *i* at column *x*.

    Notes
    -----
    The tilt angle of each order (in degrees) is logged to stdout, derived
    from the linear coefficient of the polynomial fit.

    Examples
    --------
    >>> import numpy as np
    >>> flat = np.zeros((200, 500))
    >>> flat[80:90, :] = 1000.   # simulate a single order
    >>> fits = findorder(flat, owidth=10, Norders=1)
    >>> fits[0](250)   # predicted row centre at column 250
    85.0
    """
    owidth_int      = int(owidth) # read in the expected order width from the spectrograph's specifications

    convolved_flat  = convolve1d(flat_frame, np.ones(owidth_int, dtype=float) / owidth, axis=0) # convolve the flat image to identify the orders as the peaks of the convolved data. Image is sliced in columns after this step
    # now go through the flat frame and find those orders by selecting the peak pos of the three strongest peaks in each pixel column
    
    all_peaks = []
    all_peaks_brightness = []
    for x, flat_conv_col in enumerate(convolved_flat.T):
        peaks_found, peak_params = find_peaks(flat_conv_col, distance=owidth) # find the peaks in every successive pixel column of the image (we identify them as being the spectral orders of the spectrograph)
            
        # given the number of orders to be expected, we select the N brightest peaks and then get their centroids
        brightness_mask = flat_conv_col[peaks_found] >= np.sort(flat_conv_col[peaks_found])[-Norders]
        all_peaks.append(peaks_found[brightness_mask]) # (['prominences', 'left_bases', 'right_bases', 'widths', 'width_heights', 'left_ips', 'right_ips']
        all_peaks_brightness.append(flat_conv_col[peaks_found][brightness_mask])

    all_peaks = np.array(all_peaks)
    all_peaks_brightness = np.array(all_peaks_brightness)

    order_center_fits   = []

    x_coords = np.arange(flat_frame.shape[1])

    for i in range(Norders):
        coefficients    = np.polyfit(x_coords, all_peaks[:, i], deg=order_poly_fit_degree) # coefficients for the polynomial fit function
        order_center_fits.append(np.poly1d(coefficients))
        fmt.info(f"Spectrum {i} seems to be tilted by an angle of {180/np.pi * np.arcsin(coefficients[-2]):.2f} degrees.") # take the linear term and calculate the tilt of the linear spectra against the image coordinate system.

    if output != None: # enable graphical output of the calibration flat data
        fig = plt.figure(dpi=200, figsize=(12, 9))
        ax = fig.add_subplot()
        img = ax.imshow(flat_frame)
        
        for i in range(Norders):
            ax.scatter(np.arange(flat_frame.shape[1])[::10], all_peaks[::10, i], marker="+", color="k", alpha=0.5) # plot every tenth data point of those that have been found
            ax.plot(x_coords, order_center_fits[i](x_coords)) # draw the central lines of each order
            
            ax.plot(x_coords, order_center_fits[i](x_coords) + owidth/2., c="gray", linestyle="--") # draw the upper boundary of each order
            ax.plot(x_coords, order_center_fits[i](x_coords) - owidth/2., c="gray", linestyle="--") # draw the lower boundary of each order

        ax.set_xlabel(r"$x$ pixel coordinate")
        ax.set_ylabel(r"$y$ pixel coordinate")

        plt.colorbar(img, ax=ax, shrink=0.5, pad=0.02, label="Image brightness [ADU]")
        fig.savefig(output, bbox_inches="tight")
        
        plt.close()
    
    return order_center_fits # save the coefficients of the polynomial fits to the self.orders instance an make them available for later steps




def extract_data_from_file(filename):
    """
    Load image/FITS data into a floating-point numpy array.

    Parameters
    ----------
    filename : str or Path
        Path to the file.

    Returns
    -------
    data : np.ndarray
        Floating-point numpy array.
    """

    filename = Path(filename)

    # ------------------------------------------------------------------
    # Check whether file exists
    # ------------------------------------------------------------------
    if not filename.exists():
        raise ValueError(
            f"The file '{filename}' does not seem to exist!"
        )

    # ------------------------------------------------------------------
    # File extension
    # ------------------------------------------------------------------
    suffix = filename.suffix.lower()

    # ------------------------------------------------------------------
    # FITS files
    # ------------------------------------------------------------------
    if suffix in [".fits", ".fit"]:
        data = fits.getdata(filename)

    # ------------------------------------------------------------------
    # Standard image formats
    # ------------------------------------------------------------------
    elif suffix in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:

        img = Image.open(filename)

        # Convert RGB/RGBA/etc. to grayscale
        img = img.convert("L")

        # Convert to numpy array
        data = np.array(img)

    else:
        raise ValueError(
            f"Unsupported file format: '{suffix}'"
        )

    # ------------------------------------------------------------------
    # Convert to floating point
    # ------------------------------------------------------------------
    data = np.asarray(data, dtype=float)

    return data



class DADOSobservation:
    """
    Container for a DADOS spectrograph observing run.

    Manages flat-field and wavelength-calibration data, performs order
    detection, flux extraction, and blind wavelength calibration for all
    spectral orders of the DADOS spectrograph.

    Parameters
    ----------
    flat : np.ndarray, shape (ny, nx)
        Master flat-field image used for order localisation.
    calib : np.ndarray, shape (ny, nx)
        Wavelength-calibration (arc lamp) image.
    calib_lamp : CalibModule
        Calibration lamp descriptor holding the reference line lists
        (``line_list_strong`` and ``strongest_lines``).
    camera : Camera, optional
        Camera instance describing the detector. Defaults to ``QHY268M``.
    spectrograph : Spectrograph, optional
        Spectrograph instance describing the instrument. Defaults to ``DADOS``.
    **kwargs
        Additional keyword arguments (currently unused, reserved for future
        extension).

    Attributes
    ----------
    flat : np.ndarray
    calib : np.ndarray
    camera : Camera
    spectrograph : Spectrograph
    calib_lamp : CalibModule
    orders : list of np.poly1d
        Populated by :meth:`findorder`. Empty until that method is called.
    wavelength_solutions : list
        Populated by :meth:`calibrate_wavelengths`. Contains tuples of
        ``(fit_peak_pos, fit_wavelengths, wavelength_solution)`` per order.
    Norders_found : int or None
        Number of orders found; set after :meth:`findorder` completes.
    status : str
        Human-readable pipeline status string.
    debug_mode : bool
        When ``True``, diagnostic plots are generated at each step.

    Examples
    --------
    >>> obs = DADOSobservation(
    ...     flat=flat_array,
    ...     calib=calib_array,
    ...     calib_lamp=AlpyArNe,
    ...     camera=QHY268M,
    ...     spectrograph=DADOS,
    ... )
    >>> obs.findorder()
    >>> obs.calibrate_wavelengths()
    """
    
    def __init__(self, flat, calib, calib_lamp, camera=QHY268M, spectrograph=DADOS, **kwargs): # flat frame needs to be specified. Otherwise, cannot identify spectra location

        # register the data products used to create the calbration
        
        # if the filename is provided, i.e. the data file name instead of the numpy array directly retrieve the contained data first
        if isinstance(flat, str):
            self.flat       = extract_data_from_file(flat)
        
        if isinstance(calib, str):
            self.calib      = extract_data_from_file(calib)

        else:
            self.flat           = flat
            self.calib          = calib

        # register the hardware components
        self.camera         = camera
        self.spectrograph   = spectrograph
        self.calib_lamp     = calib_lamp

        # initialize observation run attributes
        self.orders         = []
        self.wavelength_solutions = []
        self.Norders_found  = None
        self.status         = "NOT CALIBRATED"
        self.debug_mode     = False



    def export_frame(self, data_array, filename): # convenience function to enable export of a specified data array (can be any )
        """
        Write a 2-D data array to a FITS file.

        Parameters
        ----------
        data_array : np.ndarray
            The image data to be saved.
        filename : str
            Destination file path (e.g. ``"output/master_flat.fits"``).
            Existing files are silently overwritten.

        Examples
        --------
        >>> obs.export_frame(obs.flat, "master_flat.fits")
        """
        fits.writeto(data=data_array, filename=filename, overwrite=True)
    

    
    def findorder(self, output=None, show_output=False, order_detection_threshold=300, order_poly_fit_degree=3):
        """
    Locate spectral orders in the stored flat-field image.

    Wraps the module-level :func:`findorder` function using the instrument
    parameters already registered in the instance (``camera.pixel_size``,
    ``spectrograph.order_width``, ``spectrograph.Norders``). Populates
    ``self.orders`` and advances ``self.status``.

    Parameters
    ----------
    output : str, optional
        File path for the diagnostic figure. Defaults to ``""``, which
        suppresses file output.
    show_output : bool, optional
        If ``True``, the diagnostic figure is displayed interactively.
        Defaults to ``False``.
    order_detection_threshold : int, optional
        Minimum signal excess above the background required for an order to
        be detected. Defaults to ``300``.
    order_poly_fit_degree : int, optional
        Polynomial degree for fitting each order's centre position. A low
        value (2-4) is recommended to avoid overfitting. Defaults to ``3``.

    Returns
    -------
    orders : list of np.poly1d
        Polynomial fits describing the row-centre of each detected order as
        a function of pixel column. Also stored in ``self.orders``.

    Examples
    --------
    >>> obs = DADOSobservation(flat=flat_arr, calib=calib_arr, calib_lamp=AlpyArNe)
    >>> order_fits = obs.findorder(order_poly_fit_degree=3)
    >>> len(order_fits)
    3
    """
        # findorder(flat_frame, owidth, Norders, output=None, debug=False, order_detection_threshold=None, order_poly_fit_degree=3, maxiter=10)
        self.orders = findorder(self.flat, self.spectrograph.order_width / self.camera.pixel_size, self.spectrograph.Norders, debug=self.debug_mode, output=output)
        self.status = "ORDER LOCATIONS FOUND"

        return self.orders



    def extract_flux(self, image, aperture_offsets=0., aperture_height=None, order=None):
        """
    Extract one-dimensional flux profiles from spectral orders via aperture photometry.

    For each requested order, a sequence of rectangular apertures is placed
    along the order's polynomial centre trace (accounting for local tilt) and
    summed column-by-column using ``photutils.aperture_photometry``.
    Requires :meth:`findorder` to have been run first.

    Parameters
    ----------
    image : np.ndarray or str
        A 2-D science or calibration image, or a path to a FITS file.
    aperture_offsets : float, optional
        Spatial offset of the aperture centre from the order trace, in pixels,
        measured perpendicular to the dispersion axis. Useful for sky
        subtraction or nod-and-shuffle setups. Defaults to ``0.``.
    aperture_height : float or None, optional
        Height of the extraction aperture in pixels. If ``None`` (default),
        the full order width (``spectrograph.order_width / camera.pixel_size``)
        is used.
    order : int or None, optional
        Index of the single order to extract. If ``None`` (default), all
        orders are extracted.

    Returns
    -------
    signal : list of np.ndarray
        Each element is a 1-D array of extracted flux values, one per pixel
        column, for the corresponding order. Also stored in ``self.signals``.

    Raises
    ------
    Exception
        If ``self.orders`` is empty (i.e. :meth:`findorder` was not yet run).
    Exception
        If ``self.calib`` is not a valid ``np.ndarray``.

    Examples
    --------
    >>> obs.findorder()
    >>> flux = obs.extract_flux("science_target.fits")
    >>> len(flux)       # one array per order
    3
    >>> flux[0].shape   # one value per detector column
    (3056,)

    Extract only the second order with a custom aperture height:

    >>> flux_order1 = obs.extract_flux(science_arr, aperture_height=80., order=1)
    """
        
        x = np.arange(self.flat.shape[1]) # x-axis definition, reference to the bottom image axis
        if len(self.orders) == 0:
            raise Exception("[ERROR] An error occurred while trying to read the order location parameters. Make sure you ran the order identification routine before you continues with the wavelenggth calibration")
        
        theta               = [[order_fit.deriv(m=1)(x_coord) for x_coord in x] for order_fit in self.orders] # the first derivative of the polynomial fits yield the rotation at image pixel (x,y)
        y                   = [[order_fit(x_coord) for x_coord in x] for order_fit in self.orders] # y is two-dim (e.g. for DADOS (3_orders x image_width))

        apertures           = []
        
        if order != None:
            orders_to_extract = [order]
        else:
            orders_to_extract = range(self.spectrograph.Norders)
        
        if aperture_height == None:
            aperture_height = self.spectrograph.order_width / self.camera.pixel_size

        for n in orders_to_extract: 
            apertures.append(np.array([RectangularAperture(
                (
                    x[i] - np.sin(theta[n][i] * aperture_offsets),
                    y[n][i] + np.cos(theta[n][i] * aperture_offsets)
                ),
                1.,
                aperture_height,
                theta[n][i]
            ) for i in range(len(x))])) # define ALL apterure to probe through the image
        
        if not isinstance(self.calib, np.ndarray): raise Exception("Wavelength calibration image does not exist!")

        signal              = []
        if isinstance(image, str):
            data_to_read_out = fits.getdata(image)
        else:
            data_to_read_out = np.asarray(image)


        for i in range(len(orders_to_extract)): # for every order ...

            signal_data     = np.ravel([aperture_photometry(data_to_read_out, apertures[i][k])["aperture_sum"] for k in tqdm(range(len(x)))]) # read in the data from the image ... within the aperture that are already defined
            signal.append(signal_data)
        
        self.signals = signal.copy()
        self.extracted_orders = np.array(orders_to_extract)
        
        return signal
    


    def calibrate_wavelengths(self, poly_order=3, pattern_size=5, overhang=4, peak_min_dist=10, peak_prominence_threshold=None, maxNbrightest_peaks=40):
        """
        Derive blind wavelength solutions for all spectral orders.

        Extracts the arc-lamp spectrum for each order and calls
        :func:`get_wavelength_solution_blind` to match detected emission peaks
        against the reference line list in ``self.calib_lamp``. Results are
        stored in ``self.wavelength_solutions``.

        Parameters
        ----------
        poly_order : int, optional
            Degree of the polynomial wavelength solution. Defaults to ``3``.
        pattern_size : int, optional
            Number of lines in each pattern used for the initial blind match
            (``tuple_size`` in :func:`get_wavelength_solution_blind`). Values
            between 3 and 5 work best. Defaults to ``5``.
        overhang : int, optional
            Number of anchor points used when extrapolating the wavelength
            solution beyond the initially matched range. Defaults to ``4``.
        peak_min_dist : int, optional
            Minimum separation between neighbouring peaks in the arc spectrum,
            in pixels. Defaults to ``10``.
        peak_prominence_threshold : float or None, optional
            Minimum peak prominence for the arc-line finder. ``None`` disables
            thresholding. Defaults to ``None``.
        maxNbrightest_peaks : int, optional
            Maximum number of bright peaks used for the initial pattern match.
            Reducing this value speeds up the search but may decrease robustness.
            Defaults to ``40``.

        Returns
        -------
        None
            Results are stored in ``self.wavelength_solutions``, a list of length
            ``Norders``. Each element is a tuple
            ``(fit_peak_pos, fit_wavelengths, wavelength_solution)`` as returned
            by :func:`get_wavelength_solution_blind`.

        Examples
        --------
        >>> obs.findorder()
        >>> obs.calibrate_wavelengths(poly_order=3, pattern_size=4)
        >>> pix, wl, sol = obs.wavelength_solutions[0]
        >>> sol(1500)   # wavelength in Å at pixel column 1500
        5823.4
        """
        
        self.wavelength_solutions = list(np.full(self.spectrograph.Norders, None))

        for i in range(self.spectrograph.Norders):

            calib_signal = self.extract_flux(self.calib, order=i)[0]

            self.wavelength_solutions[i] = get_wavelength_solution_blind(
                calib_signal, # the calibration spectrum as a one-dimensional float array
                self.calib_lamp.strongest_lines, # list of bright calibration lines, used to find a preliminary wavelength solution
                self.calib_lamp.line_list_strong, # list of calibration lines, used to refine the preliminary wavelength solution
                self.spectrograph.wavelength_dispersion * self.camera.pixel_size, # for the DADOS+QHY268M with 1x1 binning use ~0.4
                maxNbrightest_peaks=maxNbrightest_peaks, # number of bright emission lines based on which a first estimate for a preliminary wavelength solution will be determined
                tuple_size=pattern_size, # how complex of a pattern to identify. Should be between 3-5
                order=poly_order, # order of the fitting polynomial
                peak_prominence_threshold=peak_prominence_threshold, # peak prominence threshold for the detection of peaks in the calibration signal
                peak_min_dist=peak_min_dist, # minimum distance between neighboring peaks for the peak finder
                overhang=overhang, # the number of wavelength fit points used to extrapolate a linear function
                debug=self.debug_mode, # enable or diable debug mode
                diagnostics_filename = "wavelength_calibration_diagnostics.pdf" # output filename for the diagnostic plots. If diagnostics_filename="", no file is generated. Plot will not be generated in debug=False regardless of whether it is an empty string
            )

        fmt.success("All spectrograph orders are now wavelength-calibrated!")

        self.status = "ALL WAVELENGTH CALIBRATION SUCCESSFUL"



    def get_spectrum(self, spectrum_raw_image, flat_correction=False, readout_kwargs=None, calibration_kwargs=None):
        """
        Extract wavelength-calibrated spectra from a raw science image.

        This is a high-level convenience wrapper that performs:

        1. Spectral order extraction from the wavelength-calibration frame.
        2. Blind wavelength calibration for each extracted order via
        :func:`get_wavelength_solution_blind`.
        3. Extraction of the science spectrum from ``spectrum_raw_image``.
        4. Optional flat-field correction using the stored master flat.
        5. Construction of wavelength-flux arrays for each spectral order.

        The extraction geometry and wavelength-calibration behaviour can be
        customised through the ``readout_kwargs`` and
        ``calibration_kwargs`` dictionaries.

        Parameters
        ----------
        spectrum_raw_image : np.ndarray or str
            Two-dimensional science image containing the spectrum to extract,
            or a path to a FITS file readable by ``astropy.io.fits.getdata``.
        flat_correction : bool, optional
            If ``True``, divides the extracted science flux by the extracted
            flat-field flux for each order. Defaults to ``False``.
        readout_kwargs : dict or None, optional
            Dictionary of keyword arguments forwarded to
            :meth:`extract_flux`.

            Supported keys are:

            ``aperture_offsets`` : float or None
                Spatial offset of the extraction aperture in pixels.
                ``None`` is interpreted as ``0.``.
            ``aperture_height`` : float or None
                Height of the extraction aperture in pixels.
                If ``None``, the full order width is used.
            ``order`` : int or None
                Index of a single spectral order to extract.
                If ``None``, all orders are processed.

            Defaults to ``None`` (equivalent to an empty dictionary).
        calibration_kwargs : dict or None, optional
            Dictionary of keyword arguments forwarded internally to
            :func:`get_wavelength_solution_blind`.

            Supported keys are:

            ``poly_order`` : int
                Polynomial degree of the wavelength solution.
            ``pattern_size`` : int
                Number of lines per pattern used for blind matching.
            ``overhang`` : int
                Number of anchor points used for extrapolation.
            ``peak_min_dist`` : int
                Minimum distance between neighbouring arc peaks in pixels.
            ``peak_prominence_threshold`` : float or None
                Minimum prominence threshold for peak detection.
            ``maxNbrightest_peaks`` : int
                Maximum number of bright peaks used for the initial match.

            Defaults to ``None`` (equivalent to using all default calibration
            parameters).

        Returns
        -------
        result : list of np.ndarray
            List containing one wavelength-calibrated spectrum per extracted
            order. Each element has shape ``(N, 2)`` where:

            - ``[:, 0]`` contains wavelengths in Angstrom
            - ``[:, 1]`` contains the corresponding extracted flux values

            The wavelength axis is generated from the polynomial wavelength
            solution derived for the corresponding order.

        Notes
        -----
        Wavelength calibration is recomputed every time this method is called,
        based on the currently stored calibration frame ``self.calib`` and the
        specified extraction geometry.

        If ``flat_correction=False``, the returned fluxes are uncorrected raw
        extracted counts.

        Examples
        --------
        Extract all orders from a science frame:

        >>> spectra = obs.get_spectrum("target.fits")
        >>> spectra[0].shape
        (3056, 2)

        Extract only the first order with flat correction enabled:

        >>> spec = obs.get_spectrum(
        ...     "target.fits",
        ...     flat_correction=True,
        ...     readout_kwargs={"order": 0}
        ... )

        Use custom wavelength-calibration settings:

        >>> spec = obs.get_spectrum(
        ...     "target.fits",
        ...     calibration_kwargs={
        ...         "poly_order": 4,
        ...         "pattern_size": 4,
        ...         "peak_min_dist": 8
        ...     }
        ... )

        Access wavelength and flux arrays separately:

        >>> wavelength = spectra[0][:, 0]
        >>> flux = spectra[0][:, 1]
        """

        if readout_kwargs == None:
            readout_kwargs = {}

        if calibration_kwargs == None:
            calibration_kwargs = {}



        # parse the readout dictionary
        if "aperture_offsets" in readout_kwargs:
            aperture_offsets = readout_kwargs["aperture_offsets"]

            if aperture_offsets == None:
                aperture_offsets = 0.
        else:
            aperture_offsets = 0.
        
        if "aperture_height" in readout_kwargs:
            aperture_height = readout_kwargs["aperture_height"]
        else:
            aperture_height=None

        if "order" in readout_kwargs:
            order = readout_kwargs["order"]
        else:
            order=None



        # parse the wavelength calibration dictionary
        if "poly_order" in calibration_kwargs:
            poly_order = calibration_kwargs["poly_order"]
        else:
            poly_order = 3


        if "pattern_size" in calibration_kwargs:
            pattern_size = calibration_kwargs["pattern_size"]
        else:
            pattern_size = 5


        if "overhang" in calibration_kwargs:
            overhang = calibration_kwargs["overhang"]
        else:
            overhang = 4

        
        if "peak_min_dist" in calibration_kwargs:
            peak_min_dist = calibration_kwargs["peak_min_dist"]
        else:
            peak_min_dist = 10

        
        if "peak_prominence_threshold" in calibration_kwargs:
            peak_prominence_threshold = calibration_kwargs["peak_prominence_threshold"]
        else:
            peak_prominence_threshold = None

        
        if "maxNbrightest_peaks" in calibration_kwargs:
            maxNbrightest_peaks = calibration_kwargs["maxNbrightest_peaks"]
        else:
            maxNbrightest_peaks = 40




        wavelength_calibration_fluxes = self.extract_flux(self.calib, aperture_offsets=aperture_offsets, aperture_height=aperture_height, order=order)

        wavelength_solutions = []

        for these_extracted_fluxes in wavelength_calibration_fluxes: # for each order that needs to be extracted
            wavelength_solution = get_wavelength_solution_blind(
                these_extracted_fluxes, # the calibration spectrum as a one-dimensional float array
                self.calib_lamp.strongest_lines, # list of bright calibration lines, used to find a preliminary wavelength solution
                self.calib_lamp.line_list_strong, # list of calibration lines, used to refine the preliminary wavelength solution
                self.spectrograph.wavelength_dispersion * self.camera.pixel_size, # for the DADOS+QHY268M with 1x1 binning use ~0.4
                maxNbrightest_peaks=maxNbrightest_peaks, # number of bright emission lines based on which a first estimate for a preliminary wavelength solution will be determined
                tuple_size=pattern_size, # how complex of a pattern to identify. Should be between 3-5
                order=poly_order, # order of the fitting polynomial
                peak_prominence_threshold=peak_prominence_threshold, # peak prominence threshold for the detection of peaks in the calibration signal
                peak_min_dist=peak_min_dist, # minimum distance between neighboring peaks for the peak finder
                overhang=overhang, # the number of wavelength fit points used to extrapolate a linear function
                debug=self.debug_mode, # enable or diable debug mode
                diagnostics_filename = "wavelength_calibration_diagnostics.pdf" # output filename for the diagnostic plots. If diagnostics_filename="", no file is generated. Plot will not be generated in debug=False regardless of whether it is an empty string
            )
            wavelength_solutions.append(wavelength_solution)

        spectral_fluxes               = self.extract_flux(spectrum_raw_image, aperture_offsets=aperture_offsets, aperture_height=aperture_height, order=order)
        
        if flat_correction:
            flat_fluxes               = self.extract_flux(self.flat, aperture_offsets=aperture_offsets, aperture_height=aperture_height, order=order)
        else:
            flat_fluxes               = np.array([wavelength_solution[2](range(spectral_fluxes.size)) for wavelength_solution, spectral_fluxes in zip(wavelength_solutions, spectral_fluxes)])


        result = [np.array([wavelength_solution[2](np.arange(spectral_flux.size)), spectral_flux / flat_flux]).T for wavelength_solution, spectral_flux, flat_flux in zip(wavelength_solutions, spectral_fluxes, flat_fluxes)]

        return result



class CalibModule:
    """
    Reference data container for a wavelength-calibration lamp.

    Holds two line lists for a given arc lamp: a short list of the brightest,
    most distinctive emission lines used to seed the initial pattern match, and
    a longer list of well-measured lines used to refine and extend the
    wavelength solution.

    Parameters
    ----------
    **kwargs
        Accepted keyword arguments:

        line_list_strong : np.ndarray
            Array of wavelengths (in Angstrom) of reliably detected emission
            lines suitable for refining the wavelength solution.
        strongest_lines : np.ndarray
            Small subset of the most prominent lamp lines (in Angstrom) used
            exclusively for the initial blind pattern match.

    Attributes
    ----------
    line_list_strong : np.ndarray
        Full reference line list in Angstrom.
    strongest_lines : np.ndarray
        Bright-line subset in Angstrom.

    Examples
    --------
    >>> import numpy as np
    >>> lamp = CalibModule(
    ...     line_list_strong=np.array([5852.49, 5944.83, 6074.34, 6096.16]),
    ...     strongest_lines=np.array([5852.49, 6074.34]),
    ... )
    >>> lamp.strongest_lines
    array([5852.49, 6074.34])
    """

    def __init__(self, **kwargs):

        if "line_list_strong" in kwargs:
            self.line_list_strong = kwargs["line_list_strong"]

        if "strongest_lines" in kwargs:
            self.strongest_lines = kwargs["strongest_lines"]



AlpyArNe = CalibModule(
    line_list_strong = np.array([3834.679, 3850.581, 3869.528, 3925.719, 3928.623, 3932.547, 3946.097, 3948.979, 3979.356, 3994.792, 4013.857, 4033.809, 4042.894, 4044.419, 4052.921, 4072.005, 4079.574, 4082.387, 4013.912, 4156.086, 4158.590, 4164.180, 4181.884, 4190.713, 4198.317, 4200.674, 4218.665, 4222.637, 4228.158, 4237.220, 4251.185, 4259.362, 4266.286, 4272.169, 4277.528, 4282.898, 4300.101, 4333.581, 4335.338, 4345.168, 4348.064, 4352.205, 4362.066, 4370.753, 4375.954, 4379.667, 4385.057, 4400.988, 4426.001, 4433.838, 4439.467, 4448.879, 4474.759, 4481.811, 4510.733, 4545.052, 4579.350, 4589.898, 4657.901, 4726.868, 4735.906, 4764.865, 4806.020, 4847.810, 4861.350, 4879.864, 4889.042, 4933.209, 4965.080, 5009.334, 5017.163, 5037.374, 5062.037, 5141.783, 5145.308, 5162.285, 5165.773, 5187.746, 5221.271, 5330.778, 5341.094, 5400.562, 5421.352, 5451.652, 5495.874, 5506.113, 5558.702, 5572.541, 5606.733, 5650.704, 5719.225, 5739.520, 5748.298, 5764.419, 5804.450, 5820.158, 5852.488, 5881.895, 5912.085, 5928.813, 5944.834, 5975.534, 6029.997, 6043.223, 6059.372, 6074.338, 6096.163, 6114.923, 6143.063, 6163.594, 6217.281, 6266.495, 3304.428, 6334.428, 6382.992, 6402.246, 6416.307, 6506.528, 6532.882, 6562.810, 6598.953, 6678.276, 6717.043, 6752.834]),
    strongest_lines = np.array([5852.488, 5944.834, 6029.997, 6074.338, 6096.163, 6143.063])
)





        

def get_hitmatrix(peaks, wavelengths, tuple_size, wavelength_dispersion_estimate):
    """
    Build a peak-to-wavelength hit matrix via combinatorial pattern matching.

    For every possible combination of `tuple_size` reference wavelengths,
    the function checks all combinations of `tuple_size` detected peaks for
    consistency with a constant wavelength dispersion. Consistent peak tuples
    increment the corresponding entries of the hit matrix, so that the
    highest-valued cell ``(i, j)`` indicates that peak *i* is most often
    matched to reference wavelength *j*.

    Parameters
    ----------
    peaks : array_like, shape (P,)
        Pixel positions of detected emission peaks, sorted in ascending order.
    wavelengths : array_like, shape (W,)
        Reference wavelengths in Angstrom, sorted in ascending order.
    tuple_size : int
        Number of lines per pattern. Larger values increase selectivity but
        raise computational cost as O(C(P, n) * C(W, n)). Recommended range
        is 3-5.

    Returns
    -------
    hit_matrix : np.ndarray of int, shape (P, W)
        Accumulator matrix. ``hit_matrix[i, j]`` counts how many consistent
        patterns included peak *i* matched with reference wavelength *j*.

    Raises
    ------
    Exception
        If ``len(peaks) < tuple_size`` or ``len(wavelengths) < tuple_size``.

    Notes
    -----
    A peak tuple is considered consistent with a wavelength tuple when the
    ratio of maximum to minimum implied dispersion across all tuple intervals
    is within a tight tolerance that scales with the minimum inter-peak
    spacing. An additional absolute dispersion filter (±0.2 Å/px around
    0.41 Å/px) rejects physically implausible matches.

    Examples
    --------
    >>> import numpy as np
    >>> peaks = np.array([100, 250, 410, 600, 780])
    >>> wavelengths = np.array([5852.49, 5944.83, 6074.34, 6096.16, 6143.06])
    >>> H = get_hitmatrix(peaks, wavelengths, tuple_size=3)
    >>> H.shape
    (5, 5)
    >>> H.argmax(axis=1)   # most-voted wavelength index for each peak
    array([...])
    """
    peaks = np.asarray(peaks)
    wavelengths = np.asarray(wavelengths)

    if peaks.size < tuple_size:
        raise Exception("Wavelength pattern matching is not possible. Attempted to look for patterns involving %i individual lines, but only %i peaks are provided. This is insufficient!" % (tuple_size, peaks.size))
    
    if wavelengths.size < tuple_size:
        raise Exception("Wavelength pattern matching is not possible. Attempted to look for patterns involving %i individual lines, but only %i reference wavelengths are provided. This is insufficient!" % (tuple_size, wavelengths.size))

    # generate the array of possible combinations
    tuples = np.array(list(combinations(np.arange(peaks.size), tuple_size)))

    # generate wavelength triplets
    wavelength_triplets = np.array(list(combinations(np.arange(wavelengths.size), tuple_size)))
    
    hit_matrix = np.zeros((peaks.size, wavelengths.size), dtype=int) # define an array that counts which peak is successfully associated with a given wavelength ot form a linear triplet.
    # assign tuples to wavelength triplets
    
    cum_hist_data = []
    hist_data_bins = np.arange(0.1, 2., 0.04)
    for this_wavelength_tuple in wavelength_triplets:
    
        these_wavelengths = wavelengths[this_wavelength_tuple]
    
        dist_between_peaks = np.diff(peaks[tuples], axis=1)
        
        dist_between_wavelengths = np.expand_dims(np.diff(these_wavelengths), axis=0)
        
        wavelength_dispersion = dist_between_wavelengths / dist_between_peaks
        
        mask = (np.max(wavelength_dispersion, axis=1) / np.min(wavelength_dispersion, axis=1)) < np.clip((np.min(dist_between_peaks, axis=1)+1.) / np.min(dist_between_peaks, axis=1), 1., 1.02)
        mask = mask & (np.max(np.abs(wavelength_dispersion - wavelength_dispersion_estimate), axis=1) < 0.2)
        
        for i in range(tuples.shape[1]):
            which_peaks, counter = np.unique_counts(tuples[mask, i])
            hit_matrix[which_peaks, this_wavelength_tuple[i]] += counter
        
    return hit_matrix



def Npoints_to_order(Npoints, maxorder = 4):
            """if Npoints < 6:
                return np.min([1, maxorder])
            elif Npoints < 10:
                return np.min([2, maxorder])
            elif Npoints < 15:
                return np.min([3, maxorder])
            else:
                return np.min([4, maxorder])"""
            return 1+int(Npoints / 5.)



def get_wavelength_solution_blind(
        spectrum, # the calibration spectrum as a one-dimensional float array
        first_guess_wavelengths, # list of bright calibration lines, used to find a preliminary wavelength solution
        calib_wavelengths, # list of calibration lines, used to refine the preliminary wavelength solution
        coarse_wavelength_dispersion_estimate, # for the DADOS+QHY268M with 1x1 binning use ~0.4
        maxNbrightest_peaks=40, # number of bright emission lines based on which a first estimate for a preliminary wavelength solution will be determined
        tuple_size=5, # how complex of a pattern to identify. Should be between 3-5
        order=3, # order of the fitting polynomial
        peak_prominence_threshold=None, # peak prominence threshold for the detection of peaks in the calibration signal
        peak_min_dist=10, # minimum distance between neighboring peaks for the peak finder
        overhang=4, # the number of wavelength fit points used to extrapolate a linear function
        debug=False, # enable or diable debug mode
        diagnostics_filename = "diagnostics.pdf" # output filename for the diagnostic plots. If diagnostics_filename="", no file is generated. Plot will not be generated in debug=False regardless of whether it is an empty string
    ):
    """
    Derive a polynomial wavelength solution from an arc spectrum without prior
    pixel-to-wavelength correspondence ("blind" calibration).

    The algorithm proceeds in three stages:

    1. **Initial match** — Slides a window of width proportional to the span
       of `first_guess_wavelengths` along the spectrum and calls
       :func:`get_hitmatrix` in each window to find the most likely pixel-
       wavelength associations among the brightest arc lines.
    2. **Extension** — Starting from the initial match, the solution is
       propagated up and down the full wavelength range of `calib_wavelengths`
       by fitting a rolling linear tail to the current anchor set and querying
       the nearest detected peak.
    3. **Refinement** — Three iterations of kappa-sigma clipping (3sigma) are
       applied to the polynomial fit to remove blended or mis-identified lines.

    If no valid solution is found with the original dispersion sign, the
    spectrum is flipped and the search is retried automatically.

    Parameters
    ----------
    spectrum : array_like, shape (N,)
        One-dimensional calibration (arc-lamp) spectrum, in arbitrary flux
        units.
    first_guess_wavelengths : array_like
        Small set of the brightest, most distinctive arc-lamp lines (in
        Angstrom) used exclusively for the initial pattern match.
    calib_wavelengths : array_like
        Full reference line list (in Angstrom) used to extend and refine the
        wavelength solution.
    coarse_wavelength_dispersion_estimate : float
        Approximate dispersion in Angstrom per pixel. For the DADOS
        spectrograph with the QHY268M camera at 1x1 binning use ~0.4.
    maxNbrightest_peaks : int, optional
        Maximum number of the most prominent peaks passed to the initial
        pattern matcher. Reducing this value lowers memory use but may hurt
        robustness. Defaults to ``40``.
    tuple_size : int, optional
        Pattern size (number of lines per combination) passed to
        :func:`get_hitmatrix`. Values of 3-5 are recommended. Defaults to
        ``5``.
    order : int, optional
        Polynomial degree of the final wavelength solution. Defaults to ``3``.
    peak_prominence_threshold : float or None, optional
        Minimum prominence for ``scipy.signal.find_peaks``. ``None`` disables
        the threshold (equivalent to 0). Defaults to ``None``.
    peak_min_dist : int, optional
        Minimum pixel separation between neighbouring detected peaks. Defaults
        to ``10``.
    overhang : int, optional
        Number of anchor points at the edge of the matched region used to
        fit the rolling linear extrapolant during the extension step. Defaults
        to ``4``.
    debug : bool, optional
        If ``True``, generates a four-panel diagnostic figure showing the
        pixel x wavelength associations, fit residuals, dispersion curve, and
        spectral resolution. Defaults to ``False``.
    diagnostics_filename : str, optional
        File path for the diagnostic figure when ``debug=True``. Pass ``""``
        to display interactively instead of saving. Defaults to
        ``"diagnostics.pdf"``.

    Returns
    -------
    fit_peak_pos : np.ndarray, shape (M,)
        Pixel positions of the M arc lines used in the final fit.
    fit_wavelengths : np.ndarray, shape (M,)
        Corresponding reference wavelengths in Angstrom.
    wavelength_solution : np.poly1d
        Callable polynomial mapping pixel position to wavelength in Angstrom.

    Raises
    ------
    Exception
        If no consistent wavelength solution can be found even after flipping
        the spectrum.

    Notes
    -----
    Prior to peak detection the spectrum is normalised by subtracting the
    median and dividing by the median absolute difference, then clipped to
    non-negative values. This makes `peak_prominence_threshold` largely
    scale-independent.

    Examples
    --------
    >>> pix, wl, sol = get_wavelength_solution_blind(
    ...     spectrum=arc_flux,
    ...     first_guess_wavelengths=AlpyArNe.strongest_lines,
    ...     calib_wavelengths=AlpyArNe.line_list_strong,
    ...     coarse_wavelength_dispersion_estimate=0.4,
    ...     maxNbrightest_peaks=30,
    ...     tuple_size=4,
    ...     order=3,
    ... )
    >>> sol(1024)      # wavelength in Å at pixel 1024
    5977.3
    >>> import numpy as np
    >>> residuals = wl - sol(pix)
    >>> np.std(residuals)   # RMS wavelength residual in Å
    0.08
    """
    if peak_prominence_threshold   == None:
        peak_prominence_threshold   = 0.
    
    already_tried_flipping_spectrum = False # a safe parachute so see if the spectrum needs to be flipped in order to successfully run through the wavelength calibration
    calib_wavelengths               = np.sort(np.asarray(calib_wavelengths))

    # homogenize the input and convert them to numpy arrays
    spectrum                        = np.asarray(spectrum)
    first_guess_wavelengths         = np.asarray(first_guess_wavelengths)

    # normalize the flux
    spectrum                        = np.clip((spectrum - np.median(spectrum)) / np.median(np.abs(np.diff(spectrum))), 0, np.inf)

    while True: # the outer loop to see whether the spectrum needs to be flipped. This while loop will be executed at most twice.

        # detect emission peaks
        all_peaks, all_peak_params  = find_peaks(spectrum, prominence=peak_prominence_threshold, distance=peak_min_dist)

        if all_peaks.size          <= maxNbrightest_peaks:
            fmt.warning("Only %i peaks were found which the wavelength solution can be based on. This is less than the provided `maxNbrightest_peaks`. If this is unwanted behaviour, consider lowering the `peak_prominence_threshold` setting which is currently set to %.1f." % (all_peaks.size, peak_prominence_threshold))

        if debug:
            fmt.info("Found %i peaks in the spectrum." % all_peaks.size)

        # make a subselection of all of those peaks if necessary (too many of those can confuse the routine to find a first wavelength solution)
        if all_peaks.size      > maxNbrightest_peaks:
            if debug:
                fmt.info("More peaks found than allowed maximum for first fit estimation! Proceeding with the %i most prominent ones." % maxNbrightest_peaks)
            sorted_prominence  = np.sort(all_peak_params["prominences"])
            bright_peakmask    = all_peak_params["prominences"] > sorted_prominence[-maxNbrightest_peaks]
            peaks = all_peaks[bright_peakmask]
            peak_prominences   = all_peak_params["prominences"][bright_peakmask]
        else:
            peaks = all_peaks.copy()
            peak_prominences   = all_peak_params["prominences"].copy()

        ###### make a first guess wavelength solution ######
        # take the strongest lines as defined in the first argument and build a wavelength solution based on them

        # now make a subselection of all peaks and select a window that has twice the width of the minmax range of the strongest wavelength emissions from the reference list
        window_size            = 2*int((np.max(first_guess_wavelengths) - np.min(first_guess_wavelengths)) / coarse_wavelength_dispersion_estimate)

        upper_bound            = window_size
        last_iter_flag         = False # indicate whether the current iteration in the while loop will be the last one of its kind
        all_hit_matrices       = []
        all_peaks_in_selection = []
        all_false_combinations = []

        if debug:
            fmt.info("Now scanning the spectrum for preliminary wavelength solutions ...")
        counter                = 0
        while True:
            if last_iter_flag:
                break

            if upper_bound    >= (spectrum.size):
                upper_bound    = spectrum.size
                last_iter_flag = True

            # create the subselection of peaks within the selected wavelength window
            peak_mask          = (peaks > (upper_bound - window_size)) & (peaks < upper_bound)

            try:
                all_hit_matrices.append(get_hitmatrix(peaks[peak_mask], first_guess_wavelengths, tuple_size, wavelength_dispersion_estimate=coarse_wavelength_dispersion_estimate))
                all_peaks_in_selection.append(peaks[peak_mask])
                all_false_combinations.append(comb(np.sum(peak_mask) - first_guess_wavelengths.size, first_guess_wavelengths.size))

            except Exception as e:
                if debug:
                    fmt.warning("Selection window %i (pix %i - %i): HitMatrix cannot be created (we skip this one then) due to the following exception:" % (counter, upper_bound - window_size, upper_bound))
                    fmt.warning(e)
            
            upper_bound       += window_size // 2
            counter           += 1
        
        hit_matrix_ratings_sorted = np.flip(np.argsort([np.sum(this_hit_matrix) for this_hit_matrix in all_hit_matrices])) # sort by the most promising hit matrices

        calibration_failed_flag = True
        for sorted_hit_matrix_list_id in hit_matrix_ratings_sorted:

            this_hit_matrix = all_hit_matrices[sorted_hit_matrix_list_id]

            # skip this hit matrix if there are not hits at all
            if this_hit_matrix.max() == 0:
                continue

            these_peak_pos  = all_peaks_in_selection[sorted_hit_matrix_list_id]

            # now generate the wavelength-pixel tuples:
            pixel_wavelength_pairs = []
            for this_wavelength, hit_matrix_column in zip(first_guess_wavelengths, this_hit_matrix.T):

                sorted_column = np.argsort(hit_matrix_column)
                if (hit_matrix_column[sorted_column[-2]] / hit_matrix_column[sorted_column[-1]]) > 0.5: # if there are peaks that could evnetly correspond to multiple wavelengths, discard them ...
                    continue
                else:
                    pixel_wavelength_pairs.append([these_peak_pos[sorted_column[-1]], this_wavelength])

            pixel_wavelength_pairs = np.array(pixel_wavelength_pairs)

            # at this point, a consistent first association of wavelengths and pixel peaks
            calibration_failed_flag = False
            fmt.success("The following wavelength assignments seem tentative:" + fmt.bold + "\n=================" + fmt.reset)
            for this_pixel_wavelength_pair in pixel_wavelength_pairs:
                print("%s%s%s <==> %s%.2f%s" % (fmt.rgb(1., 0.5, 0.7) + fmt.bold, ("%i" % this_pixel_wavelength_pair[0]).ljust(4), fmt.reset, fmt.rgb(*wavelength_to_rgb([this_pixel_wavelength_pair[1]])[0]) + fmt.bold, this_pixel_wavelength_pair[1], fmt.reset))
            print(fmt.bold + "=================" + fmt.reset)

            # now fit a line through this and reject clear outliers
            coef = np.polyfit(pixel_wavelength_pairs[:, 0], pixel_wavelength_pairs[:, 1], 1)
            y_fit = np.polyval(coef, pixel_wavelength_pairs[:, 0])

            residuals = np.abs(pixel_wavelength_pairs[:, 1] - y_fit)
            mask = residuals < 3. * np.std(residuals)

            coef_refined = np.polyfit(pixel_wavelength_pairs[:, 1][mask], pixel_wavelength_pairs[:, 0][mask], 1) # wavelengths --> pixel
            first_poly_fit = np.poly1d(coef_refined)


            # now assuming that the estimator works fine, use the first estimate to retrieve more lines to fit on
            peaks_tree = cKDTree(data=np.expand_dims(all_peaks, axis=1)) # initialize a location tree solver for querying different pixel locations and the closest peaks to them
            
            # with the given first estimate, now fit all the peak-wavelengths associations that are in range of the already matched peaks and wavelengths
            wavelengths_in_range = calib_wavelengths[(calib_wavelengths < np.max(pixel_wavelength_pairs[:, 1])) & (calib_wavelengths > np.min(pixel_wavelength_pairs[:, 1]))]
            distances, peak_ids = peaks_tree.query(np.expand_dims(first_poly_fit(wavelengths_in_range), axis=1))
            mask_valid = distances < peak_min_dist / 1.5 # when looking for the nearest neighbors, only mask those as valid that are closer than 2/3 the minimum allowed distance between neighboring peaks.

            # thos epeaks that could be associated with a wavelength marker, add to the list, that later the wavelength solution will be based on
            fit_wavelengths = wavelengths_in_range[mask_valid]
            fit_peak_pos    = all_peaks[peak_ids[mask_valid]]

            if debug:
                fig, axs = plt.subplots(figsize=(8, 13), nrows=4, gridspec_kw=dict(height_ratios=[3, 0.5, 1, 1], hspace=0.02), sharex=True)
                ax = axs[0]
                
                # initialize the wavelength solution diagnostic plots
                if already_tried_flipping_spectrum:
                    ax.scatter(wavelengths_in_range[mask_valid], spectrum.size - all_peaks[peak_ids[mask_valid]], color="k")
                else:
                    ax.scatter(wavelengths_in_range[mask_valid], all_peaks[peak_ids[mask_valid]], color="k")

            if debug:
                fmt.success("Possible wavelength solution found! Start to refine now ...")
                fmt.info("Adding more wavelength-peak matches ...")

            # make lists with wavelength markers that are beyond the range of the current peak-wavelength pairs
            remaining_wavelengths_up   = calib_wavelengths[calib_wavelengths > np.max(pixel_wavelength_pairs[:, 1])]
            remaining_wavelengths_down = np.flip(calib_wavelengths[calib_wavelengths < np.min(pixel_wavelength_pairs[:, 1])])

            # go the wavelength ladder upwards
            for this_wavelength in remaining_wavelengths_up:
                fit_tail = np.poly1d(np.polyfit(fit_wavelengths[-overhang:], fit_peak_pos[-overhang:], deg=1))
                # accorduing to the currect best wavelength solution, what peak pos does this corerspond to?
                estimated_peak_pos = fit_tail(this_wavelength)
                distance, id = peaks_tree.query(estimated_peak_pos)
                
                if distance > peak_min_dist / 1.5:
                    continue

                fit_wavelengths = np.append(fit_wavelengths, this_wavelength)
                fit_peak_pos    = np.append(fit_peak_pos, all_peaks[id])

                if debug:
                    print("Adding %s%s%s <==> %s%.2f%s" % (fmt.rgb(1., 0.5, 0.7) + fmt.bold, ("%i" % all_peaks[id]).ljust(4), fmt.reset, fmt.rgb(*wavelength_to_rgb([this_wavelength])[0]) + fmt.bold, this_wavelength, fmt.reset))
                    if already_tried_flipping_spectrum:
                        ax.scatter(this_wavelength, spectrum.size - all_peaks[id], s=5, color="k")
                    else:
                        ax.scatter(this_wavelength, all_peaks[id], s=5, color="k")

            # go the wavelength ladder downwards
            for this_wavelength in remaining_wavelengths_down:
                fit_tail = np.poly1d(np.polyfit(fit_wavelengths[:overhang], fit_peak_pos[:overhang], deg=1))
                # according to the current best wavelength solution, what peak pos does this correspond to?
                estimated_peak_pos = fit_tail(this_wavelength)
                distance, id = peaks_tree.query(estimated_peak_pos)
                
                # decide whether to include the point in hte calibration set or not
                if distance > peak_min_dist / 1.5:

                    # if you need to reject the point ...
                    if debug:
                        if already_tried_flipping_spectrum:
                            ax.scatter(this_wavelength, spectrum.size - all_peaks[id], marker="x", color="r")
                        else:
                            ax.scatter(this_wavelength, all_peaks[id], marker="x", color="r")
                    continue # ... skip this one!

                else:
                    # ... otherwise add the point the calibration data set!
                    fit_wavelengths = np.insert(fit_wavelengths, 0, this_wavelength)
                    fit_peak_pos    = np.insert(fit_peak_pos, 0, all_peaks[id])

                    if debug:
                        print("Adding %s%s%s <==> %s%.2f%s" % (fmt.rgb(1., 0.5, 0.7) + fmt.bold, ("%i" % all_peaks[id]).ljust(4), fmt.reset, fmt.rgb(*wavelength_to_rgb([this_wavelength])[0]) + fmt.bold, this_wavelength, fmt.reset))
                        if already_tried_flipping_spectrum:
                            ax.scatter(this_wavelength, spectrum.size - all_peaks[id], s=5, color="k")
                        else:
                            ax.scatter(this_wavelength, all_peaks[id], s=5, color="k")

            
        # if after all iterations of the selection windows still no viable wavelength solution could be estimated, raise an Exception. The wavelength solution finding routine has failed!
        if calibration_failed_flag:
            if not already_tried_flipping_spectrum:
                spectrum = np.flip(spectrum)
                fmt.warning("Finding wavelength solution with a positive wavelength dispersion failed. Attempting to find a solution with a decending trend, i. e. dLambda/dPixel < 0 ...")
                already_tried_flipping_spectrum = True
                continue
            else:
                fmt.error("Unable to find a wavelength solution. Not enough patterns on strong lines were recognized as such. Re-Check your spectrum by eye!")
                raise Exception("Failed to blindly find a wavelength solution.")
        else:
            break


    # suppose you found a good solution
    inliers = np.full(fit_peak_pos.shape, True)
    if already_tried_flipping_spectrum:

        fit_peak_pos = spectrum.size - fit_peak_pos

        wavelength_solution = np.poly1d(np.polyfit(fit_peak_pos, fit_wavelengths, deg=order))

        # apply Kappa-Sigma clipping to the final wavelength solution to clean the measurements from e.g. emission line blends
        for _ in range(3):

            # filter for outliers
            inliers = (np.abs(wavelength_solution(fit_peak_pos) - fit_wavelengths) / np.std(wavelength_solution(fit_peak_pos) - fit_wavelengths)) < 3.

            wavelength_solution = np.poly1d(np.polyfit(fit_peak_pos[inliers], fit_wavelengths[inliers], deg=order))
            fit_peak_pos    = fit_peak_pos[inliers]
            fit_wavelengths = fit_wavelengths[inliers]


        fit_peak_pos        = np.flip(fit_peak_pos[inliers])
        fit_wavelengths     = np.flip(fit_wavelengths[inliers])

    else:
        wavelength_solution = np.poly1d(np.polyfit(fit_peak_pos, fit_wavelengths, deg=order))

        for _ in range(3):

            # filter for outliers
            inliers = (np.abs(wavelength_solution(fit_peak_pos) - fit_wavelengths) / np.std(wavelength_solution(fit_peak_pos) - fit_wavelengths)) < 3.

            wavelength_solution = np.poly1d(np.polyfit(fit_peak_pos[inliers], fit_wavelengths[inliers], deg=order))
            fit_peak_pos    = fit_peak_pos[inliers]
            fit_wavelengths = fit_wavelengths[inliers]

    if debug:
        ax.plot(wavelength_solution(np.arange(spectrum.size)), np.arange(spectrum.size), color="tab:red", zorder=0)
        ax.set_ylabel(r"Sensor Coordinate $x$ [px]")

        # Plot the residuals
        ax = axs[1]
        ax.scatter(wavelength_solution(fit_peak_pos), fit_wavelengths - wavelength_solution(fit_peak_pos), zorder=10, color="k", marker="+", lw=1)
        ax.axhline(0., color="tab:red")
        ax.text(0.5, 0.95, r"$\sigma = " + "%.3g$ Å" % np.std(fit_wavelengths - wavelength_solution(fit_peak_pos), ddof=1), va="top", ha="center", transform=ax.transAxes)
        ax.set_ylabel(r"Residuals [Å]")
        ax.set_ylim(-1, 1)

        # Plot the wavelength dispersion
        ax = axs[2]
        ax.plot(wavelength_solution(fit_peak_pos), wavelength_solution.deriv(1)(fit_peak_pos), color="tab:red")
        ax.scatter(wavelength_solution((fit_peak_pos[:-1] + fit_peak_pos[1:]) / 2.), np.diff(fit_wavelengths) / np.diff(fit_peak_pos), zorder=10, color="k", marker="+", lw=1)
        ax.set_ylabel(r"$\mathrm{d}\lambda/\mathrm{d}x$ [Å/px]")

        # Plot the spectral resolution
        ax = axs[3]
        measured_peak_widths, _, _, _ = peak_widths(spectrum, all_peaks)
        ax.scatter(wavelength_solution(all_peaks), np.log10(wavelength_solution(all_peaks) / (measured_peak_widths * np.abs(wavelength_solution.deriv(1)(all_peaks)))), zorder=10, color="k", marker="+", lw=1)
        ax.set_ylabel(r"spec. Resolution $\log R$")

        ax.set_xlabel(r"Wavelength $\lambda$ [Å]")
        axs[0].set_title(get_timestamp().replace("\t", ""))
        
        if diagnostics_filename != "":
            fig.savefig(diagnostics_filename, bbox_inches="tight")
            fmt.info("Saved diagnostic PDF output to " + fmt.bold + diagnostics_filename + fmt.reset)
            plt.close()
        else:
            plt.show()

    return fit_peak_pos, fit_wavelengths, wavelength_solution