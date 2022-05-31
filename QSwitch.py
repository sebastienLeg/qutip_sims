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

    """
    All units are by default in GHz/ns
    """
    def __init__(
        self,
        EJs=None, ECs=None, gs=None,
        qubit_freqs=None, alphas=None, # specify either frequencies + anharmonicities or qubit parameters
        cutoffs=None,
        is2Q=False, # Model 2 coupled transmons instead of full QRAM module
        isCavity=[False, False, False, False]) -> None:

        self.is2Q = is2Q
        self.nqubits = 2 + 2*(not is2Q)

        if is2Q and cutoffs is None: cutoffs = [5, 5]
        elif cutoffs is None: cutoffs = [4,5,4,4]
        self.cutoffs = cutoffs

        assert gs is not None
        if np.array(gs).ndim == 0:
            gs = np.array([gs])
        self.gs = gs

        if qubit_freqs is not None and alphas is not None:
            self.qubit_freqs = qubit_freqs
            self.alphas = alphas
        else:
            assert EJs is not None and ECs is not None and gs is not None
            transmons = [scq.Transmon(EC=ECs[i], EJ=EJs[i], ng=0, ncut=110, truncated_dim=cutoffs[i]) for i in range(self.nqubits)]
    
            evals = [None]*self.nqubits
            evecs = [None]*self.nqubits
            for i in range(self.nqubits):
                evals[i], evecs[i] = transmons[i].eigensys(evals_count=cutoffs[i])
                evals[i] -= evals[i][0]
            self.qubit_freqs = [evals[i][1] for i in range(self.nqubits)]
            self.alphas = [(not isCavity[i]) * evals[i][2] - 2*evals[i][1] for i in range(self.nqubits)]

        a = qt.tensor(qt.destroy(cutoffs[0]), qt.qeye(cutoffs[1])) # source
        b = qt.tensor(qt.qeye(cutoffs[0]), qt.destroy(cutoffs[1])) # switch
        self.a_ops = [a, b]

        if not is2Q:
            a = qt.tensor(qt.destroy(cutoffs[0]), qt.qeye(cutoffs[1]), qt.qeye(cutoffs[2]), qt.qeye(cutoffs[3])) # source
            b = qt.tensor(qt.qeye(cutoffs[0]), qt.destroy(cutoffs[1]), qt.qeye(cutoffs[2]), qt.qeye(cutoffs[3])) # switch
            c = qt.tensor(qt.qeye(cutoffs[0]), qt.qeye(cutoffs[1]), qt.destroy(cutoffs[2]), qt.qeye(cutoffs[3])) # out1
            d = qt.tensor(qt.qeye(cutoffs[0]), qt.qeye(cutoffs[1]), qt.qeye(cutoffs[2]), qt.destroy(cutoffs[3])) # out2
            self.a_ops = [a, b, c, d]

        self.H_source = 2*np.pi*(self.qubit_freqs[0]*a.dag()*a + 1/2*self.alphas[0]*a.dag()*a.dag()*a*a)
        self.H_switch = 2*np.pi*(self.qubit_freqs[1]*b.dag()*b + 1/2*self.alphas[1]*b.dag()*b.dag()*b*b)
        self.H_int_01 = 2*np.pi*self.gs[0] * (a * b.dag() + a.dag() * b)
        self.H0 = self.H_source + self.H_switch
        self.H_int = self.H_int_01
        if not is2Q:
            self.H_out1 = 2*np.pi*(self.qubit_freqs[2]*c.dag()*c + 1/2*self.alphas[2]*c.dag()*c.dag()*c*c)
            self.H_out2 = 2*np.pi*(self.qubit_freqs[3]*d.dag()*d + 1/2*self.alphas[3]*d.dag()*d.dag()*d*d)
            self.H_int_12 = 2*np.pi*self.gs[1] * (b * c.dag() + b.dag() * c)
            self.H_int_13 = 2*np.pi*self.gs[2] * (b * d.dag() + b.dag() * d)
            self.H0 += self.H_out1 + self.H_out2
            self.H_int += self.H_int_12 + self.H_int_13
        self.H = self.H0 + self.H_int
        self.esys = self.H.eigenstates()

        # Time independent drive op w/o drive amp.
        # This assumes time dependence given by sin(wt).
        # If given by exp(+/-iwt), need to divide by 2.
        self.drive_ops = [
            2*np.pi*(a.dag()+a),
            2*np.pi*(b.dag()+b),
        ]
        if not is2Q:
            self.drive_ops.append(2*np.pi*(c.dag()+c))
            self.drive_ops.append(2*np.pi*(d.dag()+d))

    """
    H (not incl H_drive) in the rotating frame of a drive at wd
    H_tilde = UHU^+ - iUU^+,
    U = e^(-iw_d t (a^+ a + b^+ b + c^+ c + d^+ d))
    """
    def H_rot(self, wd, H=None):
        H_rot = H
        if H is None: H_rot = self.H
        for a in self.a_ops:
            H_rot -= wd*a.dag()*a
        return H_rot

    # ======================================= #
    # Working with states
    # ======================================= #

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
            assert evec.shape == ket_bare.shape, f'{evec.shape} {ket_bare.shape}'
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
                if self.is2Q:
                    psi_bare = self.make_bare([i1, i2])
                    overlap = np.abs(ket_dressed.overlap(psi_bare))**2
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_state = [i1, i2]
                else:
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
    Check up to n excitations that states are mapped 1:1
    (if failed, probably couplings are too strong)
    """
    def check_state_mapping(self, n):
        seen = np.zeros(self.cutoffs)
        evals, evecs = self.esys
        for evec in tqdm(evecs):
            i = tuple(self.find_bare(evec)[0])
            seen[i] += 1
            assert seen[i] == 1 or sum(i) > n, f'Mapped dressed state to {self.level_nums_to_name(i)} {seen[i]} times!!'
        print("Good enough for dressed states mappings. :)")

    def state(self, levels, esys=None):
        return self.find_dressed(self.make_bare(levels), esys=esys)[2]

    # ======================================= #
    # Getting the right pulse frequencies
    # ======================================= #
    """
    Drive frequency b/w state1 and state2 (strings representing state) 
    Stark shift from drive is ignored
    """
    def get_base_wd(self, state1, state2, keep_sign=False, esys=None):
        wd = qt.expect(self.H, self.state(state1, esys=esys)) - qt.expect(self.H, self.state(state2, esys=esys))
        if keep_sign: return wd
        return np.abs(wd)

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
    drive_qubit: 0, 1, 2, or 3
    Reference: Zeytinoglu 2015, Gideon's brute stark code
    """
    def max_overlap_H_tot_rot(self, state, amp, wd, drive_qubit=1):
        # note factor of 1/2 in front of amp since in rotating frame, assumes
        # written as a*exp(+iwt) + a.dag()*exp(-iwt)
        H_tot_rot = self.H_rot(wd) + amp/2*self.drive_ops[drive_qubit]
        return self.find_dressed(state, esys=H_tot_rot.eigenstates())[1]
    def get_wd_helper(self, state1, state2, amp, wd0, drive_qubit, wd_res=0.01, max_it=100):
        esys_rot = self.H_rot(wd0).eigenstates()
        psi1 = self.state(state1, esys=esys_rot) # dressed
        psi2 = self.state(state2, esys=esys_rot) # dressed
        plus = 1/np.sqrt(2) * (psi1 + psi2)
        minus = 1/np.sqrt(2) * (psi1 - psi2)

        # initial
        overlap_plus = self.max_overlap_H_tot_rot(plus, amp, wd0, drive_qubit=drive_qubit)
        overlap_minus = self.max_overlap_H_tot_rot(minus, amp, wd0, drive_qubit=drive_qubit)
        avg_overlap = np.mean((overlap_plus, overlap_minus))
        best_overlap = avg_overlap
        best_wd = wd0
        # print('init overlap', best_overlap)

        # trying positive shifts
        for n in range(1, max_it+1):
            wd = wd0 + n*wd_res
            esys_rot = self.H_rot(wd).eigenstates()
            psi1 = self.state(state1, esys=esys_rot) # dressed
            psi2 = self.state(state2, esys=esys_rot) # dressed
            plus = 1/np.sqrt(2) * (psi1 + psi2)
            minus = 1/np.sqrt(2) * (psi1 - psi2)
            overlap_plus = self.max_overlap_H_tot_rot(plus, amp, wd, drive_qubit=drive_qubit)
            overlap_minus = self.max_overlap_H_tot_rot(minus, amp, wd, drive_qubit=drive_qubit)
            avg_overlap = np.mean((overlap_plus, overlap_minus))
            if avg_overlap < best_overlap:
                # print('positive n', n, 'wd', wd, 'wd_res', wd_res, 'overlap', avg_overlap)
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
            overlap_plus = self.max_overlap_H_tot_rot(plus, amp, wd, drive_qubit=drive_qubit)
            overlap_minus = self.max_overlap_H_tot_rot(minus, amp, wd, drive_qubit=drive_qubit)
            avg_overlap = np.mean((overlap_plus, overlap_minus))
            if avg_overlap < best_overlap:
                # print('negative n', n, 'wd', wd, 'wd_res', wd_res, 'overlap', avg_overlap)
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
    def get_wd(self, state1, state2, amp, drive_qubit=1, verbose=True):
        wd_base = self.get_base_wd(state1, state2)
        wd = wd_base
        wd_res = 0.25
        overlap = 0
        it = 0
        while overlap < 0.99:
            if it >= 7: break
            if wd_res < 1e-6: break
            old_overlap = overlap
            wd, overlap = self.get_wd_helper(state1, state2, amp, wd0=wd, drive_qubit=drive_qubit, wd_res=wd_res)
            if verbose: print('\tnew overlap', overlap, 'wd', wd, 'wd_res', wd_res)
            if overlap == old_overlap: wd_res /= 10
            else: wd_res /= 5
            it += 1
        if verbose: print('updated drive freq (GHz) from', wd_base/2/np.pi, 'to', wd/2/np.pi)
        return wd

    """
    Pi pulse length b/w state1 and state2 (strings representing state)
    amp: freq
    """
    def get_Tpi(self, state1, state2, amp, drive_qubit=1):
        psi0 = self.state(state1)
        psi1 = self.state(state2)
        g_eff = psi0.dag() * amp * self.drive_ops[drive_qubit] * psi1 /2/np.pi
        g_eff = np.abs(g_eff[0][0][0])
        if g_eff == 0: return np.inf
        return 1/2/g_eff

    """
    Add a pi pulse between state1 and state2 immediately after the previous pulse
    t_pulse_factor multiplies t_pulse to get the final pulse length
    """
    def add_sequential_pi_pulse(self, seq, state1, state2, amp, drive_qubit=1, wd=0, phase=0, type='const', t_pulse=None, t_rise=1, t_pulse_factor=1):
        return self.add_precise_pi_pulse(seq, state1, state2, amp, drive_qubit=drive_qubit, wd=wd, phase=phase, type=type, t_pulse=t_pulse, t_rise=t_rise, t_pulse_factor=t_pulse_factor)

    """
    Add a pi pulse between state1 and state2 at time offset from the end of the 
    previous pulse
    """
    def add_precise_pi_pulse(
        self, seq:PulseSequence, state1:str, state2:str, amp,
        drive_qubit=1, wd=0, phase=0, type='const',
        t_offset=0, t_pulse=None, t_rise=1, t_pulse_factor=1, verbose=True
        ):
        if t_pulse == None: t_pulse = self.get_Tpi(state1, state2, amp=amp, drive_qubit=drive_qubit)
        t_pulse *= t_pulse_factor
        if wd == 0: wd = self.get_wd(state1, state2, amp, drive_qubit=drive_qubit, verbose=verbose)
        if type == 'const':
            seq.const_pulse(
                wd=wd,
                amp=amp,
                phase=phase,
                t_pulse=t_pulse,
                pulse_levels=(state1, state2),
                drive_qubit=drive_qubit,
                t_offset=t_offset,
                t_rise=t_rise,
                )
        return wd

    """
    Check for  clashing levels.
    tolerance: frequency tolerance for clashing levels (GHz, real freq)
    """
    def check_level_resonances(self, seq:PulseSequence, tolerance=0.050):
        pulse_names = seq.get_pulse_names(simplified=True)
        pulse_amps = seq.get_pulse_amps(simplified=True)
        good_freqs = seq.get_pulse_freqs(simplified=True)
        drive_qubits = seq.get_drive_qubits(simplified=True)

        # maps pulse to list of close-by pulses which has form (pulse name, freq)
        problem_pulses = dict()

        # Checks that any pulse a<->b is not resonant with a<->anything or b<->anything
        # (1 or 2 photon transitions), and pulse a<->b is not resonant with any of the other good freq.
        # Note that depending on the pulse sequence, it may not be a bad thing to be resonant with
        # another good freq.
        for good_pulse, good_freq in good_freqs.items():
            these_problem_pulses = dict()
            for psi0 in good_pulse:
                psi0_ids = self.level_name_to_nums(psi0)

                # Loop through all possible levels to transition to, starting from either end of the good pulse
                for i1 in range(self.cutoffs[0]):
                    for i2 in range(self.cutoffs[1]):
                        for i3 in range(self.cutoffs[2]):
                            for i4 in range(self.cutoffs[3]):
                                psi1_ids = [i1,i2,i3,i4]
                                psi1 = self.level_nums_to_name(psi1_ids)
                                pulse = (min(psi0,psi1), max(psi0,psi1)) # write in alphabetical order

                                if pulse == good_pulse: continue

                                # Don't repeat count
                                if pulse in these_problem_pulses.keys(): continue

                                # 1 or 2 photon transitions
                                n_excite = np.abs(sum(psi1_ids)-sum(psi0_ids))
                                if not 1 <= n_excite <= 2: continue
                                if n_excite != 1 and n_excite != 2: continue

                                this_freq = self.get_base_wd(*pulse, keep_sign=True)/2/np.pi
                                freq_diff_1photon = np.abs(np.abs(this_freq) - np.abs(good_freq))
                                freq_diff_2photon = np.abs(2*np.abs(this_freq) - np.abs(good_freq))
                                if min(freq_diff_1photon, freq_diff_2photon) > tolerance: continue

                                # Check if coupling too small to care
                                if self.get_Tpi(
                                    *pulse, pulse_amps[good_pulse], drive_qubit=drive_qubits[good_pulse]) > 1000: continue
                                these_problem_pulses.update({pulse:this_freq})
                
                # Compare to other good freqs
                for pulse, this_freq in good_freqs.items():
                    if pulse == good_pulse: continue
                    if pulse in these_problem_pulses.keys(): continue
                    if np.abs(np.abs(good_freq) - np.abs(this_freq)) < tolerance:
                        these_problem_pulses.update({pulse:this_freq})
            if len(these_problem_pulses.items()) > 0:
                problem_pulses.update({good_pulse:these_problem_pulses})
        return problem_pulses

    # ======================================= #
    # Assemble the H_solver with a given pulse sequence to be put into mesolve
    # ======================================= #

    def H_solver(self, seq:PulseSequence, H=None):
        if H is None: H = self.H
        H_solver = [H]
        for pulse_i, pulse_func in enumerate(seq.get_pulse_seq()):
            H_solver.append([self.drive_ops[seq.drive_qubits[pulse_i]], pulse_func])
        return H_solver

    # def H_solver_array(self, seq:PulseSequence, times):
    #     # WARNING: need to sample at short enough times for drive frequency
    #     return [self.H, 
    #     [self.drive_op, np.array([seq.pulse(t, None) for t in times])]
    #     ]

    def H_solver_str(self, seq:PulseSequence):
        H_solver = [self.H]
        pulse_str_drive_qubit = seq.get_pulse_str()
        for drive_qubit, pulse_str in enumerate(pulse_str_drive_qubit):
            H_solver.append([self.drive_ops[drive_qubit], pulse_str])
        return H_solver

    def H_solver_rot(self, seq:PulseSequence):
        H_solver = []
        for pulse_i, envelope_func in enumerate(seq.get_envelope_seq()):
            H_solver.append([
                self.H_rot(2*np.pi*seq.get_pulse_freqs()[pulse_i])\
                    + seq.get_pulse_amps()[pulse_i]/2*self.drive_ops[seq.get_drive_qubits()[pulse_i]],
                envelope_func
                ])
        return H_solver

    # ======================================= #
    # Time evolution of states
    # ======================================= #
    def evolve(self, psi0, seq:PulseSequence, times, H=None, c_ops=None, nsteps=1000, use_str_solve=False, progress=True):
        if not progress: progress = None
        if c_ops is None:
            if not use_str_solve:
                return qt.mesolve(self.H_solver(seq=seq, H=H), psi0, times, progress_bar=progress, options=qt.Options(nsteps=nsteps)).states
            return qt.mesolve(self.H_solver_str(seq), psi0, times, progress_bar=progress, options=qt.Options(nsteps=nsteps)).states
        else:
            full_result = qt.mcsolve(self.H_solver_str(seq), psi0, times, c_ops, progress_bar=progress, options=qt.Options(nsteps=nsteps))
            return np.sum(full_result.states, axis=0)/full_result.ntraj

    def evolve_rot_frame(self, psi0, seq:PulseSequence, times, c_ops=None, nsteps=1000, progress=True):
        assert c_ops == None
        if not progress: progress = None
        if c_ops is None:
            return qt.mesolve(self.H_solver_rot(seq), psi0, times, progress_bar=progress, options=qt.Options(nsteps=nsteps)).states
        else:
            pass
            # full_result = qt.mcsolve(self.H_solver_str(seq), psi0, times, c_ops, progress_bar=progress, options=qt.Options(nsteps=nsteps))
            # return np.sum(full_result.states, axis=0)/full_result.ntraj
    
    # def evolve_opt_ctrl(self, psi0, I_drives, Q_drives, qubits, times, c_ops=None, nsteps=1000, progress=True):
    #     assert c_ops == None
    #     if not progress: progress = None
    #     seq = PulseSequence()
    #     for q in qubits:
    #         seq.const_pulse(wd=2*np.pi*self.qubit_freqs[q], )
    #     if c_ops is None:
    #         return qt.mesolve(self.H_solver_opt_ctrl(I_drives, Q_drives), psi0, times, progress_bar=progress, options=qt.Options(nsteps=nsteps)).states
    #     else:
    #         pass


if __name__ == "__main__":
    EJs = [22, 21, 24, 23]
    ECs = [0.25, 0.4, 0.4, 0.28]
    gs = [0.1, 0.1, 0.1] # g01, g12, g13
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