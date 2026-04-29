import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib import rcParams
from scipy.spatial import cKDTree
from datetime import datetime
import numpy as np



# measurements
a4paper = (8.27, 11.69)
a4paper_landscape = (11.69, 8.27)

class Geometry():

    def __init__(self):

        # convert untis to inches
        self.cm      = 1. / 2.54
        self.mm      = 1. / 25.4

        self.a4paper = (8.27, 11.69)
        self.a4paper_landscape = (11.69, 8.27)

        self.aa_one_column = 88 * self.mm
        self.aa_two_column = 180 * self.mm
        self.aa_intermediate = 120 * self.mm
GEOMETRY = Geometry()

#=======================================================================================
#Nice and clean plot style.
def enable_fancyplot():
    """This function is a macro for some matplotlib settings that are used set a homogeneous plotting style throughout. Note that this function enables the LaTeX functionality in matplotlib which may let the plotting process take longer if enabled."""
    
    rcParams['lines.markeredgewidth'] = 0.
    rcParams['text.usetex'] = True
    rcParams['font.family'] = 'Computer Modern'
    #rcParams['axes.edgecolor'] = 'None'
    rcParams['xtick.top'] = True
    rcParams['xtick.bottom'] = True
    rcParams['xtick.direction'] = 'in'
    rcParams['xtick.minor.visible'] = True
    rcParams['xtick.minor.top'] = True
    rcParams['xtick.minor.bottom'] = True
    rcParams['xtick.major.top'] = True
    rcParams['xtick.major.bottom'] = True
    rcParams['ytick.left'] = True
    rcParams['ytick.right'] = True
    rcParams['ytick.direction'] = 'in'
    rcParams['ytick.minor.visible'] = True
    rcParams['ytick.minor.left'] = True
    rcParams['ytick.minor.right'] = True
    rcParams['ytick.major.left'] = True
    rcParams['ytick.major.right'] = True
    rcParams['grid.linestyle'] = '--'
    rcParams['grid.alpha'] = 0.5
    rcParams['scatter.edgecolors'] = 'None'



def flatten_list(list2d):
    """A utility function that reduces the dimansion of a list by one. Good for nested sequences

    Args:
        list2d (List): A multi-dimensional list to be squeezed.

    Returns:
        List: The flattened version of the inuput list.
    """     
    result = []
    for list1d in list2d: result += list(list1d)
    return result


BALMER_SERIES = [
    [6562.79, "H-alpha",   "#ff0000"],
    [4861.35, "H-beta",    "#00b7ff"],
    [4340.47, "H-gamma",   "#4f7fff"],
    [4101.74, "H-delta",   "#6a5cff"],
    [3970.07, "H-epsilon", "#7a4cff"],
    [3889.05, "H-zeta",    "#8a44ff"],
    [3835.39, "H-eta",     "#9640ff"],
    [3797.90, "H-theta",   "#a03cff"],
    [3770.63, "H-iota",    "#a83aff"],
    [3750.15, "H-kappa",   "#af38ff"],
    [3734.37, "H-lambda",  "#b537ff"],
    [3721.94, "H-mu",      "#ba35ff"]
]



def wavelength_to_rgb(wavelength, gamma=0.8):
    """
    Convert wavelength(s) in nm (~3800-7800) to RGB.

    Accepts:
        - scalar wavelength
        - numpy array of wavelengths

    Returns:
        numpy array (..., 3) with RGB values in [0,1]
    """

    wl = np.asarray(wavelength, dtype=float) / 10. # convert from Angstrom to nanometers
    r = np.zeros_like(wl)
    g = np.zeros_like(wl)
    b = np.zeros_like(wl)

    mask = (wl >= 380) & (wl < 440)
    r[mask] = -(wl[mask] - 440) / (440 - 380)
    b[mask] = 1.0

    mask = (wl >= 440) & (wl < 490)
    g[mask] = (wl[mask] - 440) / (490 - 440)
    b[mask] = 1.0

    mask = (wl >= 490) & (wl < 510)
    g[mask] = 1.0
    b[mask] = -(wl[mask] - 510) / (510 - 490)

    mask = (wl >= 510) & (wl < 580)
    r[mask] = (wl[mask] - 510) / (580 - 510)
    g[mask] = 1.0

    mask = (wl >= 580) & (wl < 645)
    r[mask] = 1.0
    g[mask] = -(wl[mask] - 645) / (645 - 580)

    mask = (wl >= 645) & (wl <= 780)
    r[mask] = 1.0

    factor = np.ones_like(wl)

    mask = (wl >= 380) & (wl < 420)
    factor[mask] = 0.3 + 0.7*(wl[mask] - 380)/(420 - 380)

    mask = (wl > 700) & (wl <= 780)
    factor[mask] = 0.3 + 0.7*(780 - wl[mask])/(780 - 700)

    mask = (wl < 380) | (wl > 780)
    factor[mask] = 0.0

    r = (r * factor) ** gamma
    g = (g * factor) ** gamma
    b = (b * factor) ** gamma

    rgb = np.stack([r, g, b], axis=-1)

    if np.isscalar(wavelength):
        return rgb[0]

    return rgb


def get_timestamp(human_readable=True):

    now = datetime.now()

    if human_readable:
        return now.strftime("%Y-%m-%d %H:%M:%S\t")
    else:
        return now.strftime("%Y_%m_%d_%H_%M_%S")

# Websites to look up the individual codes:
# https://stackoverflow.com/questions/4842424/list-of-ansi-color-escape-sequences
# https://www.lihaoyi.com/post/BuildyourownCommandLinewithANSIescapecodes.html
class Formatter():
    """
    A class for formatting console text with ANSI escape codes.

    Parameters
    ----------
    None

    Attributes
    ----------
    reset : str
        ANSI escape code for resetting text formatting.
    inv : str
        ANSI escape code for inverting text colors.
    rinv : str
        ANSI escape code for reverting inverted text colors.
    bold : str
        ANSI escape code for bold text.
    rbold : str
        ANSI escape code for reverting bold text.
    inf : str
        ANSI escape code for information messages.
    warning : str
        ANSI escape code for warning messages.
    error : str
        ANSI escape code for error messages.

    Methods
    -------
    warn(*messages)
        Print a warning message with appropriate formatting.
    err(*messages)
        Print an error message with appropriate formatting.
    info(*messages)
        Print an information message with appropriate formatting.

    Notes
    -----
    The class provides ANSI escape codes for common text formatting and colorization.
    It is designed for enhancing console output with colored messages.

    Examples
    --------
    >>> formatter = Formatter()
    >>> formatter.warn("This is a warning message.")
    >>> formatter.err("This is an error message.")
    >>> formatter.info("This is an information message.")
    """

    def __init__(self) -> None:
        """
        Initialize the Formatter class with ANSI escape codes for text formatting.
        """
        self.reset            = "\u001b[0m"
        self.inv              = "\u001b[47m\u001b[30m"
        self.rinv             = "\u001b[27m"
        self.bold             = "\u001b[1m"
        self.rbold            = "\u001b[22m"
        self.blink            = "\u001b[5m"
        self.rblink           = "\u001b[25m"

        self.info_color       = "\u001b[38;2;175;175;175m\u001b[1m"
        self.warning_color    = "\u001b[38;2;255;255;0m\u001b[1m"
        self.error_color      = "\u001b[38;2;255;50;50m\u001b[1m"
        self.success_color    = "\u001b[38;2;50;255;50m\u001b[1m"


    def warning(self, *messages):
        """
        Print a warning message with appropriate formatting.

        Parameters
        ----------
        *messages : tuple of str
            Variable number of message strings to be printed.
        """
        print(get_timestamp() + self.warning_color + "[WARNING]" + self.reset, *messages)
    

    def error(self, *messages):
        """
        Print an error message with appropriate formatting.

        Parameters
        ----------
        *messages : tuple of str
            Variable number of message strings to be printed.
        """
        print(get_timestamp() + self.error_color + "[ERROR]" + self.reset, *messages)
    

    def info(self, *messages):
        """
        Print an information message with appropriate formatting.

        Parameters
        ----------
        *messages : tuple of str
            Variable number of message strings to be printed.
        """
        print(get_timestamp() + self.info_color + "[INFO]" + self.reset, *messages)


    def success(self, *messages):
        """
        Print an information message with appropriate formatting.

        Parameters
        ----------
        *messages : tuple of str
            Variable number of message strings to be printed.
        """
        print(get_timestamp() + self.success_color + "[INFO]" + self.reset, *messages)
    
    
    def RGB(self, R, G, B):
        return "\033[38;2;%i;%i;%im" % (R, G, B)

    def rgb(self, R, G, B):
            return "\033[38;2;%i;%i;%im" % (int(255.*R), int(255.*G), int(255.*B))
    
    def RGB_bg(self, R, G, B):
        return "\033[48;2;%i;%i;%im" % (R, G, B)

    def rgb_bg(self, R, G, B):
            return "\033[48;2;%i;%i;%im" % (int(255.*R), int(255.*G), int(255.*B))

    def sm2color(self, sm : ScalarMappable, value, bg = False):
        mpl_color = sm.to_rgba(value)
        if bg:
            return self.rgb_bg(mpl_color[0], mpl_color[1], mpl_color[2])
        else:
            return self.rgb(mpl_color[0], mpl_color[1], mpl_color[2])



def rebin_spectrum(wavelengths_old, fluxes_old, wavelengths_new, errs_old=None, keep_edges=False, fill_value=np.nan):
    """
    This is a helper routine that rebins a spectrum given the the spectral axis, fluxes, and desired new bins to which it should be rebinned. It is constructed in a way that is conserves the total flux and works for an inhimigeneous (re-)sampled wavelength range.

    Args:

        wavelengths_old (np.ndarray): The old wavelength stencils from the input spectrum
        
        fluxes_old (np.ndarray):      The spectrum's fluxes before rebinning them
        
        wavelenths_new (np.ndarray):  The new spectrum's bins
        
        errs_old (np.ndarray, optional): The flux errors (for now assumed to be symmetric) before rebinning. (defaults to: None)
        
        keep_edges (bool, optional):    Decide, whether to keep the edges that are otherwise filled with zeros. (defaults to: False)

        fill_value (float, optional): If keep_edges is enabled, this makrs the fill value for the flux data that needs to be filled with a placeholder value (defaults to: np.nan)

        returns wavelengths_new, fluxes_new, np.zeros_like(fluxes_new)
    """
    
    fluxes_new = np.zeros_like(wavelengths_new)

    def find_bin_edges_widths(data):

        bin_edges = np.empty(data.shape[0] + 1)
        bin_edges[1:-1] = (data[:-1] + data[1:]) / 2. # compute the midpoint-stencils

        # treat the boundaries by simply extrapolating the second to first and second to last bin
        bin_edges[0] = 2. * bin_edges[1] - bin_edges[2]
        bin_edges[-1] = 2. * bin_edges[-2] - bin_edges[-3]
        
        # compute all the bin widths
        bin_widths = (bin_edges[1:] - bin_edges[:-1])

        return bin_edges, bin_widths

    bin_edges_new, bin_widths_new = find_bin_edges_widths(wavelengths_new)
    bin_edges_old, bin_widths_old = find_bin_edges_widths(wavelengths_old)

    ## cross-match the old and new bin-edges
    # refine the sampling such that the spectral axis is sampled

    centroid_tree_old = cKDTree(np.expand_dims(wavelengths_old, axis=1))
    edge_tree_old = cKDTree(np.expand_dims(bin_edges_old, axis=1))

    distances, ids = edge_tree_old.query(
        np.expand_dims(bin_edges_new, axis=1),
        k=1
    )

    # check if two individual flux stencils from the non-rebinned spectrum are equally far away
    centroid_dists, centroid_ids = centroid_tree_old.query(
        np.expand_dims(bin_edges_new, axis=1),
        k=1
    )

    centroid_dists = np.array(centroid_dists)
    centroid_ids = np.array(centroid_ids)

    fluxes_extended = np.concatenate((fluxes_old, [fluxes_old[-1]])) # repeat the last element of the fluxes array to make the dimensions match

    directions = np.sign(bin_edges_old[ids] - bin_edges_new)

    fluxes_new = []
    for n in range(wavelengths_new.size):

        flux_new = np.sum(fluxes_old[ids[n]:ids[n+1]] * bin_widths_old[ids[n]:ids[n+1]])
        flux_new += fluxes_extended[centroid_ids[n]] * distances[n] * directions[n]
        flux_new -= fluxes_extended[centroid_ids[n+1]] * distances[n+1] * directions[n+1]
        fluxes_new.append(flux_new)
    
    fluxes_new = np.array(fluxes_new) / bin_widths_new

    if keep_edges:
        # assign a fill value to the wavelengths that are out of range from the initial spectrum to take samples from
        
        # fill the range where the new wavelength stencils are below the defined range
        fluxes_new = np.where(
            bin_edges_new[1:] >= bin_edges_old[0],
            fluxes_new,
            np.full_like(fluxes_new, fill_value)
        )

        # fill the range where the new wavelength stencils are above the defined range
        fluxes_new = np.where(
            bin_edges_new[:-1] <= bin_edges_old[-1],
            fluxes_new,
            np.full_like(fluxes_new, fill_value)
        )
    
    else:
        # if the auto-filled bin should not be kept, delete them ...
        wavelength_mask = (bin_edges_new[1:] >= bin_edges_old[0]) & (bin_edges_new[:-1] <= bin_edges_old[-1])

        wavelengths_new = wavelengths_new[wavelength_mask]
        fluxes_new      = fluxes_new[wavelength_mask]


    return wavelengths_new, fluxes_new, np.zeros_like(fluxes_new) # the third return value in the tuple is a placeholder value for the flux uncertainties