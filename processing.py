import numpy as np
import scipy.optimize as opt
from collections import defaultdict


BATCHES = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 51, 52, 
                    53, 54, 55, 56, 57, 58, 59, 60, 65, 70])
CATEGORIES = np.array([1, 1, 2, 3, 5, 4, 2, 3, 4, 5, 6, 3, 4, 6,  
                       2, 5, 4, 5, 6, 3, 6, 6])

############################################################################
#                     ALIGNING CHANNELS AND SIGNALS                        #
############################################################################
# These parameters were obtained through linear regression. Please see
# the corresponding notebook.
def align(signal):
    OFFSETS = [2.180783, 2.221318, 2.221318, 2.221318, 4.42923651, 2.221318]
    SLOPE = 0.8107
    for i, (start, end) in enumerate(zip(BATCHES[:-1], BATCHES[1:])):
        start = start*100_000
        end = end*100_000
        c = CATEGORIES[i] - 1
        signal[start:end] = OFFSETS[c] + SLOPE * signal[start:end]
    return signal


############################################################################
#                     POWER LINE INTERFERENCE REMOVAL                      #
############################################################################
# We will fit a single sine wave (A * sin(w * t + p)) on all categories, 
# except for category 5, which is a two-component sine wave:
# (A1 * sin(w1 * t + p1) + A2 * sin(w2 * t + p2))
def sin_1(t, A, w, p):
    return A * np.sin(w*t + p)

def sin_4(t, A1, w1, p1, A2, w2, p2, A3, w3, p3, A4, w4, p4):
    return A1 * np.sin(w1*t + p1) + A2 * np.sin(w2*t + p2) + \
    	   A3 * np.sin(w3*t + p3) + A4 * np.sin(w4*t + p4)

def get_guess(tt, yy, components=1):
    """Initialize amplitudes at 0.053, phases at 0 and get the frequencies 
    corresponding to the highest strengths."""
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    freqs = ff[np.argsort(Fyy[1:])[::-1] + 1]
    result = set()
    for f in freqs:
        result.add(abs(f))
        if len(result) == components:
            break
    guess = []
    for f in result:
        guess += [0.053, min(0.0315, max(0.031, 2*np.pi*f)), 0]
    
    return guess

def fit_sin1(tt, yy):
    '''Fit sine wave to the input time sequence'''
    # Taken from: https://stackoverflow.com/questions/16716302
    guess = np.array(get_guess(tt, yy))
    popt, pcov = opt.curve_fit(sin_1, tt, yy, p0=guess, maxfev=10000)
    return {"parameters": popt, "fitfunc": lambda t: sin_1(t, *popt)}

# We hard-code the parameters for the sin fits of our cat 5 data. This
# part can be removed but will result in a slightly worse score (but still
# a solid gold medal position). During the competition, the power line 
# interference was very carefully curated at all times to ensure that we 
# did not mess up the public or private LB part.
CAT5_SIN_PARAMS = [
	# 1st train batch (0-100K)
	[0.057, 0.0313961337824152, -1.9582981618981699, 
	 0.052000000000000005, 0.0313373879661514, -0.894787657310209, 
	 0.00312544415186912, 0.71604979030357, 1.4740988377606499, 
	 0.0, 0.764755919779924, 0.0829734133838624],
	# 1st train batch (100K-200K)
	[0.057, 0.0313095771558318, -0.35820841715589197, 
	 0.0569999999999976, 0.031379394855977204, -0.230748302134919,
	 0.00253486686822344, 0.715851225706187, -0.026387038081517898, 
	 0.0, 0.7465297968654719, -0.0894642790306936],
	# 1st train batch (200K-300K)
	[0.05263655157016801, 0.0314659272351229, 0.721245080052644, 
	 0.052000000000000005, 0.0313174115779501, -1.37524249139897, 
	 0.00226771045170094, 0.715873941177297, 0.56031287795554, 
	 0.0019583453631626, 0.7286561468259489, -1.02618438535636],
	# 1st train batch (300K-400K)
	[0.057, 0.0313245064928946, 2.06319152201257, 
	 0.052000000003234, 0.031446893905081, -1.6084985645729901, 
	 0.00204345494144802, 0.71565222942307, -1.89474551042615, 
	 0.0026604410809482797, 0.7284881442918499, 0.7151709270600919],
	# 1st train batch (400K-500K)
	[0.056679791179581704, 0.031382822206142295, -3.5348478221412, 
	 0.052000000000000005, 0.0313178273242294, -1.00966029805514, 
	 0.0016164838108070302, 0.716048844816851, -2.59390758029503e-05, 
	 0.00210240831407528, 0.727720466174145, 0.706750747651938],
	# 2nd train batch (0-100K)
	[0.052000000000000005, 0.0313670674841525, -1.37434901558769, 
	 0.0520000000021022, 0.0313783360445586, 2.92821291855846, 
	 0.00247837389454667, 0.7158583949172861, -0.40179295728015496, 
	 0.00437980418941088, 0.728349991781816, 3.73281587410011],
	# 2nd train batch (100K-200K)
	[0.0520000000000012, 0.0313514090723553, -0.7592491943882651, 
	 0.052000000000000005, 0.031413273914979195, 0.935896221804064, 
	 0.00206902568326639, 0.715918789583516, -1.35448757805627, 
	 0.00166972502016302, 0.728623972795409, -1.4221758956342],
	# 2nd train batch (200K-300K)
	[0.052000000000287205, 0.0313326971169099, 1.4075870088394502, 
	 0.0554068634006138, 0.0313842512482485, 0.048745427259011403, 
	 0.0025172803889046804, 0.7155805029559871, 0.5406817908985491, 
	 0.00161604404966698, 0.729016882437157, 1.8120270717328],
	# 2nd train batch (300K-400K)
	[0.052000000000000005, 0.031409876692340605, -0.7456245686622709, 
	 0.0520000000001529, 0.0313705465395396, 0.936863731408621, 
	 0.00277763601864694, 0.71570598138205, -1.44532928551161, 
	 0.00211063918551229, 0.728488636873564, -1.68280443918254],
	# 2nd train batch (400K-500K)
	[0.057, 0.0313882617049079, 1.03970925443823, 
	 0.0569999997793795, 0.031387205017098, -4.4219546144679995, 
	 0.0033378429937860198, 0.7159949659289311, -0.592157376889125, 
	 0.0, 0.7701378258839879, -0.578583201201017],
	# Public LB Part
	[0.052000000000000005, 0.031340030736758, 2.03496434423533, 
	 0.057, 0.0313644869979618, 5.994735891834329, 
	 0.0030110230573384, 0.71556629357372, 3.95735883922847, 
	 0.0, 0.718902128677972, 0.0990162017704937],
	# Private LB Part
	[0.052000000000000005, 0.0313962479935369, 2.25668278709478, 
	 0.052000000000000005, 0.0313859582890873, -4.05125563031034, 
	 0.00235983507549681, 0.715410020428299, -0.0953467752653618, 
	 0.0, 0.7255387616378449, -1.89409831367182]
]

def fit_sin(signal, channels, c, b):
    # This noise calculation worked better than a simple signal - channels
    noise = signal - (0.2 * channels + 0.8 * np.round(channels))
    offset = np.median(noise)
    noise = noise - offset
    noise = np.clip(noise, -0.8, 0.8)
    if c != 5:
        results = fit_sin1(np.arange(len(noise)), noise)
    else:
        results = {'fitfunc': lambda t: sin_4(t, *CAT5_SIN_PARAMS[b])}

    return signal - results['fitfunc'](np.arange(len(signal))) - offset

def remove_power_line(signal, predictions):
    """Iterate over segments of 100K and remove the power line interference"""
    count_per_cat = defaultdict(int)
    for i, (start, end) in enumerate(zip(BATCHES[:-1], BATCHES[1:])):
        c = CATEGORIES[i]
        for k in range(end - start):
            sub_start = (start + k) * 100_000
            sub_end = (start + k + 1) * 100_000
            signal[sub_start:sub_end] = fit_sin(signal[sub_start:sub_end], 
                                                predictions[sub_start:sub_end], 
                                                c, count_per_cat[c])
            count_per_cat[c] += 1
    return signal


############################################################################
#                        UNSUPERVISED THRESHOLDING                         #
############################################################################
# We use an unsupervised technique in order to map the continuous Y's that are 
# calculated by the forward-backward algorithm to a discrete number of open
# channels. This is done by taking signal values that are very close to
# their rounded value (x - round(x) < threshold), as the probability is very
# likely that the number of open channels corresponds to x there. We then
# calculate percentages of each open channels and extrapolate to entire batch.

def optimize_thres_unsupervised(pred, sig):
    catbatches = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 
                  4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 
                  3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 2, 3, 5, 1, 4, 3, 4, 5, 2, 
                  5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    sY_all = [[0.26]*12,
              [0.26]*12,
              [0.26]*12,
              [0.22]*12, #0.22
              [0.26]*12,
              [0.26]*12,
             ]
    L = 100000
    Y = pred.copy()
    Thres = {}
    Yopt = pred.copy()

    for b in range(70):
        Thres[b] = np.zeros(12)
        Thres[b][0] = -99
        Thres[b][-1] = 99
        poscat = range(L*b, L*(b+1))
        catbatch = int(catbatches[b])
        sY = sY_all[catbatch]

        Yloc = Y[poscat]
        floc = sig[poscat]

        adaptive_sY = np.array([sY[int(np.round(item))] for item in floc])
        floc2 = floc[np.abs(floc-np.round(floc)) - adaptive_sY < 0]

        for i in range(10):
            ni = len(floc2[np.round(floc2)<=i])
            ni2 = np.round(ni*len(floc)/ max(1, len(floc2))).astype(int)
            Ys = np.concatenate([np.sort(floc), [19]])
            Thres[b][i+1] = 0.5*(Ys[max(0,ni2)]+Ys[min(len(Ys)-1,ni2)])

        for i in range(11):
            Yloc[(Yloc>=Thres[b][i])&(Yloc<Thres[b][i+1])] = i
        Yopt[poscat] = Yloc

    return Yopt, Thres