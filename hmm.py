import numpy as np

def calculate_matrix(Ptran, states, number_processes):
    """Extends a HMM, corresponding to a binary Markov Process,
    (i.e. 0 or 1 open channels) to model up until K open channels
    by assuming K independent binary Markov processes."""
    # Fill in diagonals such that each row sums to 1
    for i in range(Ptran.shape[0]):
        Ptran[i, i] = 1 - np.sum(Ptran[i, :])

    n0 = len(states)
    new = Ptran.copy()
    new_states = [(x,) for x in range(n0)]
    for process in range(1, number_processes):
        # We expand our current transition matrix (that models up to 
        # `process` number of separate processes) its' dimensions by n0. We 
        # basically add another possible state transition for a new process.
        nc = new.shape[0]
        Ptran_temp = np.zeros((n0*nc, n0*nc))
        temp_states = []
        for i in range(n0):
            temp_states.extend([s + (i,) for s in new_states])
            for j in range(n0):
                # We add i -> j as our final transition
                Ptran_temp[i*nc:(i+1)*nc, j*nc:(j+1)*nc] = Ptran[i][j] * new
              
        # We now group similar processes together to reduce our matrix. 
        # E.g. (1, 2, 3) is the same as (2, 3, 1)
        new_states = sorted(list(set([tuple(sorted(x)) for x in temp_states])))
        new = np.zeros((len(new_states), len(new_states)))
        for i in range(len(new_states)):
            ix_i = [k for k, x in enumerate(temp_states) 
                    if tuple(sorted(x)) == new_states[i]]
            for j in range(len(new_states)):
                ix_j = [k for k, x in enumerate(temp_states) 
                        if tuple(sorted(x)) == new_states[j]]
                new[i, j] = np.sum(Ptran_temp[ix_i, :][:, ix_j])
                new[i, j] /= len(ix_i)
    
    new_channels = []
    for s in new_states:
        new_channels.append(sum([states[x] for x in s]))
    new_channels = np.array(new_channels)
        
    return new, new_channels

def get_Psig(signal, States, kexp):
    """The provided States correspond to numbers of open channels,
    this calculates the PDF of a Gaussian, with the exception of some
    constants, assuming means equal to the open channels and a tunable
    variance (kexp)"""
    Psig = np.zeros((len(signal), len(States)))
    for i in range(len(Psig)):
        Psig[i] = np.exp((-(signal[i] - States)**2)/(kexp))
    return Psig

def forward(Psig, Ptran, etat_in=None, coef=1, normalize=True):
    """Custom forward-backward algorithm. This function is also used for the
    backward pass by reversing Psig and transposing Ptran. This custom 
    function is both faster and slightly more accurate than, for example,
    hmmlearn. It introduces memory (defined by coef), by taking in to account
    the calculated probabilities from a previous pass (etat_in)."""
    if etat_in is None: etat_in = np.ones(Psig.shape)/Psig.shape[1]
    alpha = np.zeros(Psig.shape) # len(sig) x n_state
    etat = np.zeros(Psig.shape) # len(sig) x n_state
    
    etat[0] = etat_in[0]
    alpha[0] = etat_in[0]
    if normalize: 
        alpha[0] = etat_in[0]*Psig[0]
        alpha[0]/=alpha[0].sum()

    for j in range(1, Psig.shape[0]):
        etat[j] = alpha[j-1]@Ptran
        if normalize: etat[j] /= etat[j].sum()
        etat[j] = (etat[j]**coef) * ((etat_in[j])**(1-coef))
        if normalize: etat[j] /= etat[j].sum()
        alpha[j] = etat[j]  * Psig[j]
        alpha[j] /= alpha[j].sum()
    return alpha, etat