import numpy as np
import scqubits as scq
import qutip as qt

from PulseSequence import PulseSequence
scq.settings.PROGRESSBAR_DISABLED = True


hbar = 1
h = hbar*2*np.pi
qe = 1

# Unit conversions
MHz = 10.0**(-3)
GHz = 1.0
kHz = 10.0**(-6)
us = 10.0**3
ns = 1.0

class QSwitch():
    lvl_name_to_num = dict(g=0, e=1, f=2, h=3, j=4)
    lvl_num_to_name = ['g', 'e', 'f', 'h', 'j']

    def __init__(
        self,
        EJs, ECs, gs,
        cutoffs=[4,5,4,4],
        isCavity=[False, False, False, False]) -> None:

        self.EJs = EJs
        self.ECs = ECs
        self.gs = gs
        self.cutoffs = cutoffs

        transmon1 = scq.Transmon(EC=ECs[0], EJ=EJs[0], ng=0, ncut=110, truncated_dim=cutoffs[0])
        transmon2 = scq.Transmon(EC=ECs[1], EJ=EJs[1], ng=0, ncut=110, truncated_dim=cutoffs[1])
        transmon3 = scq.Transmon(EC=ECs[2], EJ=EJs[2], ng=0, ncut=110, truncated_dim=cutoffs[2])
        transmon4 = scq.Transmon(EC=ECs[3], EJ=EJs[3], ng=0, ncut=110, truncated_dim=cutoffs[3])
    
        evals1, evecs1 = transmon1.eigensys(evals_count=cutoffs[0])
        evals2, evecs2 = transmon2.eigensys(evals_count=cutoffs[1])
        evals3, evecs3 = transmon3.eigensys(evals_count=cutoffs[2])
        evals4, evecs4 = transmon4.eigensys(evals_count=cutoffs[3])
    
        evals1 -= evals1[0]
        evals2 -= evals2[0]
        evals3 -= evals3[0]
        evals4 -= evals4[0]

        alpha1 = alpha2 = alpha3 = 0
        if not isCavity[0]: alpha1 = evals1[2]-2*evals1[1]
        if not isCavity[1]: alpha2 = evals2[2]-2*evals2[1]
        if not isCavity[2]: alpha3 = evals3[2]-2*evals3[1]
        if not isCavity[3]: alpha4 = evals4[2]-2*evals4[1]

        a = qt.tensor(qt.destroy(cutoffs[0]), qt.qeye(cutoffs[1]), qt.qeye(cutoffs[2]), qt.qeye(cutoffs[3])) # source
        b = qt.tensor(qt.qeye(cutoffs[0]), qt.destroy(cutoffs[1]), qt.qeye(cutoffs[2]), qt.qeye(cutoffs[3])) # switch
        c = qt.tensor(qt.qeye(cutoffs[0]), qt.qeye(cutoffs[1]), qt.destroy(cutoffs[2]), qt.qeye(cutoffs[3])) # out1
        d = qt.tensor(qt.qeye(cutoffs[0]), qt.qeye(cutoffs[1]), qt.qeye(cutoffs[2]), qt.destroy(cutoffs[3])) # out2

        H_source    = 2*np.pi*(evals1[1]*a.dag()*a + 1/2*alpha1*a.dag()*a*(a.dag()*a - 1))
        H_switch    = 2*np.pi*(evals2[1]*b.dag()*b + 1/2*alpha2*b.dag()*b*(b.dag()*b - 1))
        H_out1      = 2*np.pi*(evals3[1]*c.dag()*c + 1/2*alpha3*c.dag()*c*(c.dag()*c - 1))
        H_out2      = 2*np.pi*(evals4[1]*d.dag()*d + 1/2*alpha4*d.dag()*d*(d.dag()*d - 1))
        H_int_12 = 2*np.pi*gs[0] * (a * b.dag() + a.dag() * b)
        H_int_23 = 2*np.pi*gs[1] * (b * c.dag() + b.dag() * c)
        H_int_24 = 2*np.pi*gs[2] * (b * d.dag() + b.dag() * d)

        self.H = H_source + H_switch + H_out1 + H_out2 + H_int_12 + H_int_23 + H_int_24
        self.esys = self.H.eigenstates()
        self.H_drive = 2*np.pi* 1/2 * (b.dag()+b)


    def level_name_to_nums(self, name):
        state = []
        for l in name:
            state.append(self.lvl_name_to_num[l])
        return state

    def level_nums_to_name(self, nums):
        state = ''
        for n in nums:
            state += self.lvl_num_to_name[n]
        return state

    """
    Map bare states of each transmon to dressed states in combined system
    """
    def find_dressed(self, ket_bare):
        evals, evecs = self.esys
        best_overlap = 0
        best_state = -1
        for n, evec in enumerate(evecs):
            assert evec.shape == ket_bare.shape
            overlap = np.abs(ket_bare.overlap(evec))
            if overlap > best_overlap:
                best_overlap = overlap
                best_state = n
        # print(best_state)
        return best_state, evecs[best_state]

    """
    Map dressed states to bare states
    """
    def find_bare(self, ket_dressed):
        best_overlap = 0
        best_state = None
        for i1 in range(self.cutoffs[0]):
            for i2 in range(self.cutoffs[1]):
                for i3 in range(self.cutoffs[2]):
                    for i4 in range(self.cutoffs[3]):
                        psi_bare = self.make_bare([i1, i2, i3, i4])
                        overlap = np.abs(ket_dressed.overlap(psi_bare))
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_state = [i1, i2, i3, i4]
        return best_state, self.make_bare(best_state)

    """
    levels can be either array of numbers (states) or string
    """
    def make_bare(self, levels):
        prod = []
        # print(levels)
        if isinstance(levels, str):
            levels = self.level_name_to_nums(levels)
        for q, lvl in enumerate(levels):
            prod.append(qt.basis(self.cutoffs[q], lvl))
        return qt.tensor(*prod)

    def state(self, levels):
        return self.find_dressed(self.make_bare(levels))[1]

    """
    Drive frequency b/w state1 and state2 (strings representing state) 
    """
    def get_wd(self, state1, state2):
        return qt.expect(self.H, self.state(state1)) - qt.expect(self.H, self.state(state2))

    """
    Pi pulse length b/w state1 and state2 (strings representing state)
    """
    def get_Tpi(self, state1, state2, amp):
        psi0 = self.state(state1)
        psi1 = self.state(state2)
        g_eff = psi0.dag() * amp * self.H_drive * psi1 /2/np.pi
        g_eff = np.abs(g_eff[0][0][0])
        if g_eff == 0: return -1
        return 1/2/g_eff

    """
    Add a pi pulse between state1 and state2 immediately after the previous pulse
    """
    def add_sequential_pi_pulse(self, seq, state1, state2, amp, t_pulse=None):
        if t_pulse == None: t_pulse = self.get_Tpi(state1, state2, amp=amp)
        seq.wait(seq.const_pulse(wd=self.get_wd(state1, state2), amp=amp, t_pulse=t_pulse))