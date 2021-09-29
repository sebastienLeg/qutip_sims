import numpy as np
import scqubits as scq
import qutip as qt

from tqdm import tqdm

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

        alpha1 = alpha2 = alpha3 = alpha4 = 0
        if not isCavity[0]: alpha1 = evals1[2]-2*evals1[1]
        if not isCavity[1]: alpha2 = evals2[2]-2*evals2[1]
        if not isCavity[2]: alpha3 = evals3[2]-2*evals3[1]
        if not isCavity[3]: alpha4 = evals4[2]-2*evals4[1]

        self.qubit_freqs = [evals1[1], evals2[1], evals3[1], evals4[1]]
        self.alphas = [alpha1, alpha2, alpha3, alpha4]

        a = qt.tensor(qt.destroy(cutoffs[0]), qt.qeye(cutoffs[1]), qt.qeye(cutoffs[2]), qt.qeye(cutoffs[3])) # source
        b = qt.tensor(qt.qeye(cutoffs[0]), qt.destroy(cutoffs[1]), qt.qeye(cutoffs[2]), qt.qeye(cutoffs[3])) # switch
        c = qt.tensor(qt.qeye(cutoffs[0]), qt.qeye(cutoffs[1]), qt.destroy(cutoffs[2]), qt.qeye(cutoffs[3])) # out1
        d = qt.tensor(qt.qeye(cutoffs[0]), qt.qeye(cutoffs[1]), qt.qeye(cutoffs[2]), qt.destroy(cutoffs[3])) # out2

        self.a = a
        self.b = b
        self.c = c
        self.d = d

        H_source    = 2*np.pi*(evals1[1]*a.dag()*a + 1/2*alpha1*a.dag()*a.dag()*a*a)
        H_switch    = 2*np.pi*(evals2[1]*b.dag()*b + 1/2*alpha2*b.dag()*b.dag()*b*b)
        H_out1      = 2*np.pi*(evals3[1]*c.dag()*c + 1/2*alpha3*c.dag()*c.dag()*c*c)
        H_out2      = 2*np.pi*(evals4[1]*d.dag()*d + 1/2*alpha4*d.dag()*d.dag()*d*d)
        H_int_12 = 2*np.pi*gs[0] * (a * b.dag() + a.dag() * b)
        H_int_23 = 2*np.pi*gs[1] * (b * c.dag() + b.dag() * c)
        H_int_24 = 2*np.pi*gs[2] * (b * d.dag() + b.dag() * d)

        self.H = H_source + H_switch + H_out1 + H_out2 + H_int_12 + H_int_23 + H_int_24
        self.esys = self.H.eigenstates()

        # Time independent drive op w/o drive amp.
        # This assumes time dependence given by sin(wt).
        # If given by exp(+/-iwt), need to divide by 2.
        self.drive_op = 2*np.pi* (b.dag()+b)

    """
    H (not incl H_drive) in the rotating frame of a drive at wd
    H_tilde = UHU^\dag - iU\dot{U}^\dag,
    U = e^{-iw_d t (a^\dag a + b^\dag b + c^\dag c + d^\dag d)}
    """
    def H_rot(self, wd):
        a, b, c, d = (self.a, self.b, self.c, self.d)
        return self.H - wd*(a.dag()*a + b.dag()*b + c.dag()*c + d.dag()*d)

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
    def find_dressed(self, ket_bare, esys=None):
        if esys == None: esys = self.esys
        evals, evecs = esys
        best_overlap = 0
        best_state = -1
        for n, evec in enumerate(evecs):
            assert evec.shape == ket_bare.shape
            overlap = np.abs(ket_bare.overlap(evec))**2
            if overlap > best_overlap:
                best_overlap = overlap
                best_state = n
        # print(best_state)
        # print('final best overlap', best_overlap)
        return best_state, best_overlap, evecs[best_state]

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
                        overlap = np.abs(ket_dressed.overlap(psi_bare))**2
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

    """
    Check up to 3 excitations that states are mapped 1:1
    (if failed, probably couplings are too strong)
    """
    def check_state_mapping(self, n):
        seen = np.zeros(self.cutoffs)
        evals, evecs = self.esys
        for evec in tqdm(evecs):
            i = tuple(self.find_bare(evec)[0])
            seen[i] += 1
            if seen[i] != 1 and sum(i) <= n:
                print(f'Mapped dressed state to {self.level_nums_to_name(i)} {seen[i]} times!!')
                return False
        print("Good enough for dressed states mappings.")
        return True

    def state(self, levels, esys=None):
        return self.find_dressed(self.make_bare(levels), esys=esys)[2]

    """
    Drive frequency b/w state1 and state2 (strings representing state) 
    Stark shift from drive is ignored
    """
    def get_base_wd(self, state1, state2, esys=None):
        return qt.expect(self.H, self.state(state1, esys=esys)) - qt.expect(self.H, self.state(state2, esys=esys))

    """
    Fine-tuned drive frequency taking into account stark shift from drive
    Idea: adjust drive frequency until (state1 +/- state2) in the dressed
        basis of H have the max mean overlap with the estates of the 
        full H w/ drive
    state1, state2: strings or array of ints representing bare states
    amp: drive amp (freq)
    wd_res: resolution of wd sweeping (angular freq)
    base_shift: base stark shift to try
    max_it: max number of iterations when searching for shifts
    Reference: Zeytinoglu 2015, Gideon's brute stark code
    """
    def max_overlap_H_tot_rot(self, state, amp, wd):
        # note factor of 1/2 in front of amp since in rotating frame, assumes
        # written as a*exp(+iwt) + a.dag()*exp(-iwt)
        H_tot_rot = self.H_rot(wd) + amp/2 *self.drive_op
        return self.find_dressed(state, esys=H_tot_rot.eigenstates())[1]
    def get_wd_helper(self, state1, state2, amp, wd0, wd_res=0.01, max_it=100):
        psi1 = self.state(state1) # dressed
        psi2 = self.state(state2) # dressed
        plus = 1/np.sqrt(2) * (psi1 + psi2)
        minus = 1/np.sqrt(2) * (psi1 - psi2)

        # initial
        overlap_plus = self.max_overlap_H_tot_rot(plus, amp, wd0)
        overlap_minus = self.max_overlap_H_tot_rot(minus, amp, wd0)
        avg_overlap = np.mean((overlap_plus, overlap_minus))
        best_overlap = avg_overlap
        best_wd = wd0
        # print('init overlap', overlap_plus, overlap_minus)

        # trying positive shifts
        for n in range(1, max_it+1):
            wd = wd0 + n*wd_res
            esys_rot = self.H_rot(wd).eigenstates()
            psi1 = self.state(state1, esys=esys_rot) # dressed
            psi2 = self.state(state2, esys=esys_rot) # dressed
            plus = 1/np.sqrt(2) * (psi1 + psi2)
            minus = 1/np.sqrt(2) * (psi1 - psi2)
            overlap_plus = self.max_overlap_H_tot_rot(plus, amp, wd)
            overlap_minus = self.max_overlap_H_tot_rot(minus, amp, wd)
            avg_overlap = np.mean((overlap_plus, overlap_minus))
            # print('positive n', n, 'wd', wd, 'wd_res', wd_res, 'overlap', overlap_plus, overlap_minus)
            if avg_overlap < best_overlap:
                break
            else:
                best_overlap = avg_overlap
                best_wd = wd
        if n == max_it: print("Too many iterations, try lower resolution!")

        # trying negative shifts
        for n in range(1, max_it+1):
            wd = wd0 - n*wd_res
            esys_rot = self.H_rot(wd).eigenstates()
            plus = 1/np.sqrt(2) * (psi1 + psi2)
            minus = 1/np.sqrt(2) * (psi1 - psi2)
            overlap_plus = self.max_overlap_H_tot_rot(plus, amp, wd)
            overlap_minus = self.max_overlap_H_tot_rot(minus, amp, wd)
            avg_overlap = np.mean((overlap_plus, overlap_minus))
            # print('negative n', n, 'wd', wd, 'wd_res', wd_res, 'overlap', avg_overlap)
            if avg_overlap < best_overlap:
                break
            else:
                best_overlap = avg_overlap
                best_wd = wd
        if n == max_it: print("Too many iterations, try lower resolution!")

        return best_wd, best_overlap

    """
    Fine-tuned drive frequency taking into account stark shift from drive,
    analyzed with 3 different steps of increasingly small resolution
    Reference: Gideon's brute stark code
    """
    def get_wd(self, state1, state2, amp):
        # very important to take abs val!
        wd_base = np.abs(self.get_base_wd(state1, state2))
        wd = wd_base
        wd_res = 0.25
        overlap = 0
        for it in range(4):
            if overlap > 0.99: break
            wd, overlap = self.get_wd_helper(state1, state2, amp, wd0=wd, wd_res=wd_res/(5**it))
            print('\tnew overlap', overlap, 'wd', wd)
        print('updated wd from', wd_base/2/np.pi, 'to', wd/2/np.pi)
        return wd

    """
    Pi pulse length b/w state1 and state2 (strings representing state)
    """
    def get_Tpi(self, state1, state2, amp):
        psi0 = self.state(state1)
        psi1 = self.state(state2)
        g_eff = psi0.dag() * amp * self.drive_op * psi1 /2/np.pi
        g_eff = np.abs(g_eff[0][0][0])
        if g_eff == 0: return -1
        return 1/2/g_eff

    """
    Add a pi pulse between state1 and state2 immediately after the previous pulse
    t_pulse_factor multiplies t_pulse to get the final pulse length
    """
    def add_sequential_pi_pulse(self, seq, state1, state2, amp, t_pulse=None, t_rise=1, t_pulse_factor=1):
        self.add_const_pi_pulse(seq, state1, state2, amp, t_pulse=t_pulse, t_rise=t_rise, t_pulse_factor=t_pulse_factor)

    """
    Add a pi pulse between state1 and state2 at time offset from the beginning of the 
    previous pulse
    """
    def add_const_pi_pulse(self, seq, state1, state2, amp, wd=0, t_offset=0, t_pulse=None, t_rise=1, t_pulse_factor=1):
        if t_pulse == None: t_pulse = self.get_Tpi(state1, state2, amp=amp)
        t_pulse *= t_pulse_factor
        if wd == 0: wd = self.get_wd(state1, state2, amp)
        seq.const_pulse(
            wd=wd,
            amp=amp,
            t_pulse=t_pulse,
            t_start=-t_offset,
            t_rise=t_rise,
            )

    """
    Assemble the H_solver with a given pulse sequence to be put into mesolve
    """ 
    def H_solver(self, seq:PulseSequence):
        return [self.H, [self.drive_op, seq.pulse]]
    def H_solver_array(self, seq:PulseSequence, times):
        # WARNING: need to sample at short enough times for drive frequency
        return [self.H, 
        [self.drive_op, np.array([seq.pulse(t, None) for t in times])]
        ]
    def H_solver_str(self, seq:PulseSequence):
        return [self.H, [self.drive_op, seq.get_pulse_str()]]


if __name__ == "__main__":
    EJs = [22, 21, 24, 23]
    ECs = [0.25, 0.4, 0.4, 0.28]
    gs = [0.1, 0.1, 0.1] # g12, g23, g24
    cutoffs = [4, 5, 4, 4]
    isCavity = [False, False, False, False]

    qram = QSwitch(
        EJs=EJs,
        ECs=ECs,
        gs=gs,
        cutoffs=cutoffs,
        isCavity=isCavity,
    )

    qubit_freqs = qram.qubit_freqs
    alphas = qram.alphas
    print(qubit_freqs[0], qubit_freqs[1], qubit_freqs[2], qubit_freqs[3])
    print(alphas[0], alphas[1], alphas[2], alphas[3])


    from PulseSequence import PulseSequence
    times = np.linspace(0, 200, 200)
    seq = PulseSequence(start_time=0)
    qram.add_sequential_pi_pulse(seq, 'eggg', 'gfgg', amp=0.12)