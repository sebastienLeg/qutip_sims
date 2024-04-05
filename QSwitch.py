import numpy as np
import scipy as sp
import scqubits as scq
import qutip as qt
from copy import deepcopy

from tqdm import tqdm

from PulseSequence import PulseSequence, gaussian
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

    def transmon_f0n(self, n, EC, EJ):
        return np.sqrt(8*EC*EJ)*n - EC/12 * (6*n**2 + 6*n + 3) + EC/4
    
    def transmon_fge(self, EC, EJ):
        return self.transmon_f0n(1, EC, EJ) - self.transmon_f0n(0, EC, EJ)
    
    def transmon_fef(self, EC, EJ):
        return self.transmon_f0n(2, EC, EJ) - self.transmon_f0n(1, EC, EJ)
    
    def transmon_alpha(self, EC, EJ):
        return self.transmon_fef(EC, EJ) - self.transmon_fge(EC, EJ)
    

    """
    All units are by default in GHz/ns
    """
    def __init__(
        self,
        EJs=None, ECs=None, gs=None, # gs=matrix with size corresponding to number of qubits
        qubit_freqs=None, alphas=None, # specify either frequencies + anharmonicities or qubit parameters
        useZZs=False, ZZs=None, # specify qubit freqs and ZZ shifts to construct H instead, aka dispersive hamiltonian
        cutoffs=None,
        isCavity=[False, False, False, False]) -> None:

        assert cutoffs is not None
        self.useZZs = useZZs
        self.cutoffs = cutoffs
        self.nqubits = len(cutoffs)

        self.alphas = np.array(alphas)

        if self.useZZs:
            assert qubit_freqs is not None
            assert ZZs is not None
            self.qubit_freqs = np.array(qubit_freqs) # w*adag*a = w*sigmaZ/2
            self.ZZs = np.array(ZZs)
        else:
            assert gs is not None
            gs = np.array(gs)
            assert len(gs.shape) == 2, 'Need a 2D matrix for gs!'
            assert gs.shape[0] == gs.shape[1], 'Need a square matrix for gs!'
            assert np.allclose(gs, gs.T), 'Need a symmetric matrix for gs!'
            self.gs = gs

            if qubit_freqs is not None and alphas is not None:
                self.qubit_freqs = np.array(qubit_freqs)

            else:
                assert EJs is not None and ECs is not None and gs is not None
    
                # self.qubit_freqs = [self.transmon_fge(ECs[i], EJs[i]) for i in range(self.nqubits)]
                # self.alphas = [(not isCavity[i])*(self.transmon_alpha(ECs[i], EJs[i])) for i in range(self.nqubits)]
                
                transmons = [scq.Transmon(EC=ECs[i], EJ=EJs[i], ng=0, ncut=110, truncated_dim=cutoffs[i]) for i in range(self.nqubits)]
    
                evals = [None]*self.nqubits
                evecs = [None]*self.nqubits
                for i in range(self.nqubits):
                    evals[i], evecs[i] = transmons[i].eigensys(evals_count=cutoffs[i])
                    evals[i] -= evals[i][0]
                self.qubit_freqs = [evals[i][1] for i in range(self.nqubits)]
                self.alphas = [(not isCavity[i]) * evals[i][2] - 2*evals[i][1] for i in range(self.nqubits)]

        self.a_ops = [None]*self.nqubits
        for q in range(self.nqubits):
            aq = [qt.qeye(cutoffs[i]) if i != q else qt.destroy(cutoffs[i]) for i in range(self.nqubits)]
            aq = qt.tensor(*aq)
            self.a_ops[q] = aq

        self.H0 = 0*self.a_ops[0]

        self.H_Qis = []
        for q in range(self.nqubits):
            a = self.a_ops[q]
            H_q = 2*np.pi*(self.qubit_freqs[q]*a.dag()*a + 1/2*self.alphas[q]*a.dag()*a.dag()*a*a)
            self.H_Qis.append(H_q)
            self.H0 += H_q

        if not self.useZZs:
            self.H_int = 0*self.H0
            for i in range(self.nqubits):
                for j in range(i+1, self.nqubits):
                    a = self.a_ops[i]
                    b = self.a_ops[j]
                    self.H_int += 2*np.pi*self.gs[i, j] * (a * b.dag() + a.dag() * b)
        else: # use ZZ shift values: what is the adjustment to qi when qj is in e?
            ZZs = (self.ZZs + np.transpose(self.ZZs))/2 # average over J_ml and J_lm
            self.H_int = 0*self.H0
            for i in range(self.nqubits):
                for j in range(i+1, self.nqubits):
                    a = self.a_ops[i]
                    b = self.a_ops[j]
                    self.H_int += 2*np.pi*ZZs[i, j] * (a.dag() * a * b.dag() * b)
        self.H = self.H0 + self.H_int
        self.esys = self.H.eigenstates()

        # Time independent drive op w/o drive amp.
        # This assumes time dependence given by sin(wt).
        # If given by exp(+/-iwt), need to divide by 2.
        self.drive_ops = []
        for q in range(self.nqubits):
            self.drive_ops.append(2*np.pi*(self.a_ops[q].dag() + self.a_ops[q]))


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

    def H_rot_qubits(self, qubit_frame_freqs=None, H=None):
        H_rot = H
        if qubit_frame_freqs is None: qubit_frame_freqs = self.qubit_freqs
        if H is None: H_rot = self.H
        for q, a in enumerate(self.a_ops):
            H_rot -= 2*np.pi*qubit_frame_freqs[q]*a.dag()*a
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

        # Scale by product with bare evec to remove phase
        best_evec = evecs[best_state] / ket_bare.overlap(evecs[best_state])
        best_evec = best_evec.unit()
        return best_state, best_overlap, best_evec

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


    def get_ZZ(self, qA, qB, qA_state='e', qB_state='e'): # how much does qA_state shift when qB is in qB_state?
        gstate = 'g'*self.nqubits
        estate = gstate[:qA] + qA_state + gstate[qA+1:]
        qfreq = self.get_base_wd(gstate, estate, keep_sign=True)/2/np.pi

        gestate = gstate[:qB] + qB_state + gstate[qB+1:]
        eestate = gestate[:qA] + qA_state + gestate[qA+1:]
        qfreq_shift = self.get_base_wd(gestate, eestate, keep_sign=True)/2/np.pi

        # print(abs(qfreq), abs(qfreq_shift))

        # print(estate, self.get_base_wd(gstate, estate)/2/np.pi)
        # print(gestate, self.get_base_wd(gstate, gestate)/2/np.pi)
        # print(eestate, self.get_base_wd(gstate, eestate)/2/np.pi)
        # return abs(qfreq_shift) - abs(qfreq)
        return qfreq_shift - qfreq


    def get_ZZ_matrix(self):
        ZZ_mat = np.zeros((self.nqubits, self.nqubits))
        for q_spec in range(self.nqubits):
            for q_pi in range( self.nqubits):
                if q_pi == q_spec: continue
                ZZ_mat[q_spec, q_pi] = self.get_ZZ(q_spec, q_pi)
        return ZZ_mat


    def state(self, levels, esys=None):
        return self.find_dressed(self.make_bare(levels), esys=esys)[2]

    # ======================================= #
    # Getting the right pulse frequencies
    # ======================================= #
    """
    Drive frequency from state1 to state2 (strings representing state) 
    Stark shift from drive is ignored
    """
    def get_base_wd(self, state1, state2, keep_sign=False, esys=None, **kwargs):
        wd = qt.expect(self.H, self.state(state2, esys=esys)) - qt.expect(self.H, self.state(state1, esys=esys))
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

    def get_wd_helper(self, state1, state2, amp, wd0, drive_qubit, wd_res=0.01, max_it=100, **kwargs):
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
    def get_wd(self, state1, state2, amp, drive_qubit=1, verbose=True, **kwargs):
        wd_base = self.get_base_wd(state1, state2, **kwargs)
        wd = wd_base
        wd_res = 0.25
        overlap = 0
        it = 0
        while overlap < 0.99:
            if it >= 7: break
            if wd_res < 1e-6: break
            old_overlap = overlap
            wd, overlap = self.get_wd_helper(state1, state2, amp, wd0=wd, drive_qubit=drive_qubit, wd_res=wd_res, **kwargs)
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
    def get_Tpi(self, state1, state2, amp, drive_qubit=1, pihalf=False, type='const', phi_ext=None, esys=None, **kwargs):
        if esys is None: esys = self.esys
        if phi_ext is not None:
            coupler_H0, H0, H = self.get_H_at_phi_ext(phi_ext)
            esys = H.eigenstates()
        
        psi0 = self.state(state1, esys=esys)
        psi1 = self.state(state2, esys=esys)
        g_eff = psi0.dag() * amp * self.drive_ops[drive_qubit] * psi1 /2/np.pi
        g_eff = np.abs(g_eff[0][0][0])
        if g_eff == 0: return np.inf
        # In general, the formula for this is Tpi = 1/2/(g_eff * R), where
        # R: integrate the pulse shape over the length of the pulse, letting the maximum amplitude be 1, and divide by the characteristic timescale that will go into the pulse as the time length parameter - idea is that R is the scaling parameter between the area of this pulse shape and the constant pulse that adjusts how much longer the pulse needs to be
        if type=='const': return 1/2/g_eff
        elif type=='gauss':
            if 'sigma_n' not in kwargs or kwargs['sigma_n'] is None: sigma_n = 4
            else: sigma_n = kwargs['sigma_n']
            tpi = 1/2 / (g_eff * np.sqrt(2*np.pi) * sp.special.erf(sigma_n/2 / np.sqrt(2)))
        elif type=='flat_top':
            if 'sigma_n' not in kwargs or kwargs['sigma_n'] is None: sigma_n = 2
            else: sigma_n = kwargs['sigma_n']
            if 't_ramp' not in kwargs or kwargs['t_ramp'] is None: t_ramp = 15
            else: t_ramp = kwargs['t_ramp']
            t_ramp_sigma = t_ramp/sigma_n
            tpi = (1 + 2*g_eff*np.sqrt(2*np.pi)*t_ramp_sigma*sp.special.erf(sigma_n/np.sqrt(2))) / (2*g_eff)
        elif type == 'adiabatic':
            beta = kwargs['beta']
            tpi = 1/2 / (g_eff * np.arctan(np.sinh(beta)) / beta)
        else: assert False, 'Pulse type not implemented'
        return tpi 

    """
    Add a pi pulse between state1 and state2 immediately after the previous pulse
    t_pulse_factor multiplies t_pulse to get the final pulse length
    """
    def add_sequential_pi_pulse(self, seq, state1, state2, amp, pihalf=False, drive_qubit=1, wd=None, phase=0, type='const', t_offset=0, t_pulse=None, t_rise=None, t_pulse_factor=1, **kwargs):
        return self.add_precise_pi_pulse(seq, state1, state2, amp, pihalf=pihalf, drive_qubit=drive_qubit, wd=wd, phase=phase, type=type, t_offset=t_offset, t_pulse=t_pulse, t_rise=t_rise, t_pulse_factor=t_pulse_factor, **kwargs)

    """
    Add a pi pulse between state1 and state2 at time offset from the end of the 
    previous pulse
    Setting amp is None is used as a flag to just increment wait time corresponding to the shape specified, not play a pulse
    """
    def add_precise_pi_pulse(
        self, seq:PulseSequence, state1:str, state2:str, amp, pihalf=False,
        drive_qubit=1, wd=None, phase=0, type='const',
        t_offset=0, t_pulse=None, t_pulse_factor=1, verbose=True, **kwargs
        ):
        if amp is None: assert t_pulse is not None
        else:
            if t_pulse is None:
                t_pulse = self.get_Tpi(state1, state2, amp=amp, drive_qubit=drive_qubit, type=type, **kwargs)
                if pihalf: t_pulse /= 2
            t_pulse *= t_pulse_factor
            if wd is None: wd = self.get_wd(state1, state2, amp, drive_qubit=drive_qubit, verbose=verbose, **kwargs)
        if type == 'const':
            if 't_rise' not in kwargs.keys() or kwargs['t_rise'] is None: kwargs['t_rise'] = 1
            seq.const_pulse(
                wd=wd,
                amp=amp,
                phase=phase,
                t_pulse=t_pulse,
                pulse_levels=(state1, state2),
                drive_qubit=drive_qubit,
                t_offset=t_offset,
                t_rise=kwargs['t_rise'],
                )
        elif type == 'gauss':
            if 'sigma_n' not in kwargs.keys() or kwargs['sigma_n'] is None: kwargs['sigma_n'] = 4
            seq.gaussian_pulse(
                wd=wd,
                amp=amp,
                phase=phase,
                t_pulse_sigma=t_pulse,
                pulse_levels=(state1, state2),
                drive_qubit=drive_qubit,
                t_offset=t_offset,
                sigma_n=kwargs['sigma_n'],
                )
        elif type == 'flat_top':
            if 'sigma_n' not in kwargs.keys(): kwargs['sigma_n'] = 2
            if 't_rise' not in kwargs.keys() or kwargs['t_rise'] is None: kwargs['t_rise'] = 15
            seq.flat_top_pulse(
                wd=wd,
                amp=amp,
                phase=phase,
                t_pulse=t_pulse,
                pulse_levels=(state1, state2),
                drive_qubit=drive_qubit,
                t_offset=t_offset,
                t_rise=kwargs['t_rise'],
                sigma_n=kwargs['sigma_n'],
                )
        elif type == 'adiabatic':
            seq.adiabatic_pulse(
                wd=wd,
                amp=amp,
                mu=kwargs['mu'],
                beta=kwargs['beta'],
                period=t_pulse,
                pulse_levels=(state1, state2),
                drive_qubit=drive_qubit,
                t_offset=t_offset,
            )
        else: assert False, 'Pulse type not implemented'
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
            if seq.get_pulse_amps()[pulse_i] is not None:
                H_solver.append([self.drive_ops[seq.drive_qubits[pulse_i]], pulse_func])
        return H_solver

    def H_solver_array(self, seq:PulseSequence, times, H=None):
        # WARNING: need to sample at short enough times for drive frequency
        if H is None: H = self.H
        H_solver = [H]
        for pulse_i, pulse_func in enumerate(seq.get_pulse_seq()):
            if seq.get_pulse_amps()[pulse_i] is not None:
                waveform = np.array([pulse_func(t, None) for t in times])
                H_solver.append([self.drive_ops[seq.drive_qubits[pulse_i]], waveform])
        return H_solver

    def H_solver_str(self, seq:PulseSequence):
        assert False, 'this has not been debugged in so long, no guarantee of working as expected'
        H_solver = [self.H]
        pulse_str_drive_qubit = seq.get_pulse_str()
        for drive_qubit, pulse_str in enumerate(pulse_str_drive_qubit):
            H_solver.append([self.drive_ops[drive_qubit], pulse_str])
        return H_solver

    # In rotating frame of qubits
    def H_solver_rot(self, seq:PulseSequence):
        assert self.useZZs, "this only works with dispersive hamiltonian currently"

        # H_solver = [self.H_rot_qubits()]
        H_solver = [None]
        qubit_frame_freqs = [0]*self.nqubits
        for pulse_i, envelope_func in enumerate(seq.get_envelope_seq()):
            if seq.get_pulse_amps()[pulse_i] is not None:
                q = seq.get_drive_qubits()[pulse_i]
                fd = seq.get_pulse_freqs()[pulse_i]
                if not qubit_frame_freqs[q]: qubit_frame_freqs[q] = fd
                else: assert fd == qubit_frame_freqs[q], f'Must have just a single rotating frame! {fd} does not match first freq {qubit_frame_freqs[q]}'
                if hasattr(envelope_func, "__len__"):
                    envelope_func_I = envelope_func[0]
                    envelope_func_Q = envelope_func[1]
                    a_op_rot_I = self.a_ops[seq.drive_qubits[pulse_i]] * np.exp(-1j*seq.get_pulse_phases()[pulse_i])
                    a_op_rot_Q = self.a_ops[seq.drive_qubits[pulse_i]] * np.exp(-1j*seq.get_pulse_phases()[pulse_i]-1j*np.pi/2)
                    H_solver.append([seq.get_pulse_amps()[pulse_i]/2*(a_op_rot_I.dag() + a_op_rot_I)*2*np.pi, envelope_func_I])
                    H_solver.append([seq.get_pulse_amps()[pulse_i]/2*(a_op_rot_Q.dag() + a_op_rot_Q)*2*np.pi, envelope_func_Q])
                else:
                    a_op_rot = self.a_ops[seq.get_drive_qubits()[pulse_i]] * np.exp(1j*seq.get_pulse_phases()[pulse_i])
                    H_solver.append([
                        seq.get_pulse_amps()[pulse_i]/2*(a_op_rot + a_op_rot.dag())*2*np.pi,
                        envelope_func
                        ])
        # qubit_frame_freqs = self.qubit_freqs
        H_solver[0] = self.H_rot_qubits(qubit_frame_freqs=qubit_frame_freqs)
        return H_solver

    # In rotating frame of drive (angular)
    def H_solver_rot_wd(self, seq:PulseSequence, wframe):
        # H_solver = [self.H_rot_qubits()]
        H_solver = [None]
        for pulse_i, envelope_func in enumerate(seq.get_envelope_seq()):
            if seq.get_pulse_amps()[pulse_i] is not None:
                q = seq.get_drive_qubits()[pulse_i]
                fd = seq.get_pulse_freqs()[pulse_i]
                assert np.isclose(2*np.pi*fd, wframe), f'Your hamiltonian will not be correct with this function if your rotating frame is not the same as your drive frequency! {fd} does not match frame freq {wframe/2/np.pi}'
                if hasattr(envelope_func, "__len__"):
                    envelope_func_I = envelope_func[0]
                    envelope_func_Q = envelope_func[1]
                    a_op_rot_I = self.a_ops[seq.drive_qubits[pulse_i]] * np.exp(-1j*seq.get_pulse_phases()[pulse_i])
                    a_op_rot_Q = self.a_ops[seq.drive_qubits[pulse_i]] * np.exp(-1j*seq.get_pulse_phases()[pulse_i]-1j*np.pi/2)
                    H_solver.append([seq.get_pulse_amps()[pulse_i]/2*(a_op_rot_I.dag() + a_op_rot_I)*2*np.pi, envelope_func_I])
                    H_solver.append([seq.get_pulse_amps()[pulse_i]/2*(a_op_rot_Q.dag() + a_op_rot_Q)*2*np.pi, envelope_func_Q])
                else:
                    a_op_rot = self.a_ops[seq.get_drive_qubits()[pulse_i]] * np.exp(1j*seq.get_pulse_phases()[pulse_i])
                    H_solver.append([
                        seq.get_pulse_amps()[pulse_i]/2*(a_op_rot + a_op_rot.dag())*2*np.pi,
                        envelope_func
                        ])
        H_solver[0] = self.H_rot(wd=wframe)
        import h5py
        h = H_solver[0].full()
        with h5py.File("H_rot_wd0.hdf5", "w") as f:
            dset = f.create_dataset("H_rot_wd0", shape=h.shape, dtype=h.dtype, data=h)
        return H_solver

    # def H_solver_rot(self, seq:PulseSequence):
    #     H_solver = [self.H_rot(2*np.pi*seq.get_pulse_freqs()[pulse_i])]
    #     for pulse_i, envelope_func in enumerate(seq.get_envelope_seq()):
    #         H_solver.append([
    #             seq.get_pulse_amps()[pulse_i]/2*(
    #                 self.a_ops[seq.get_drive_qubits()[pulse_i]]*np.exp(1j*seq.get_pulse_phases()[pulse_i]) +
    #                 self.a_ops[seq.get_drive_qubits()[pulse_i]].dag()*np.exp(-1j*seq.get_pulse_phases()[pulse_i])),
    #             envelope_func
    #             ])
    #     return H_solver

    # def H_solver_unrotate(self, seq:PulseSequence, H=None):
    #     if H is None: H = self.H
    #     H_solver = [H]
    #     for pulse_i, pulse_func in enumerate(seq.get_pulse_seq()):
    #         H_solver.append([self.a_ops[seq.drive_qubits[pulse_i]].dag(), pulse_func])
    #         H_solver.append([self.a_ops[seq.drive_qubits[pulse_i]], lambda t,args=None: np.conj(pulse_func(t,args))])
    #     return H_solver

    # ======================================= #
    # Time evolution of states
    # Note having max_step set is important - if there is a long wait time at the beginning sometimes this can cause the solver
    # to miss the later pulses!
    # ======================================= #
    
    def evolve(self, psi0, seq:PulseSequence, times, H=None, c_ops=None, nsteps=1000, max_step=0.1, use_str_solve=False, progress=True):
        if not progress: progress = None
        if c_ops is None:
            if not use_str_solve:
                return qt.mesolve(self.H_solver(seq=seq, H=H), psi0, times, progress_bar=progress, options=qt.Options(nsteps=nsteps, max_step=max_step)).states
            return qt.mesolve(self.H_solver_str(seq), psi0, times, progress_bar=progress, options=qt.Options(nsteps=nsteps, max_step=max_step)).states
        else:
            full_result = qt.mcsolve(self.H_solver_str(seq), psi0, times, c_ops, progress_bar=progress, options=qt.Options(nsteps=nsteps))
            return np.sum(full_result.states, axis=0)/full_result.ntraj

    def evolve_array(self, psi0, seq:PulseSequence, times, H=None, c_ops=None, nsteps=1000, max_step=0.1, use_str_solve=False, progress=True):
        if not progress: progress = None
        if c_ops is None:
            return qt.mesolve(self.H_solver_array(seq=seq, times=times, H=H), psi0, times, progress_bar=progress, options=qt.Options(nsteps=nsteps, max_step=max_step)).states
        else:
            full_result = qt.mcsolve(self.H_solver_array(seq, times=times, H=H), psi0, times, c_ops, progress_bar=progress, options=qt.Options(nsteps=nsteps, max_step=max_step))
            return np.sum(full_result.states, axis=0)/full_result.ntraj

    def evolve_rot_frame(self, psi0, seq:PulseSequence, times, c_ops=None, nsteps=1000, max_step=None, progress=True):
        assert c_ops == None
        if not progress: progress = None
        if c_ops is None:
            return qt.mesolve(self.H_solver_rot(seq), psi0, times, progress_bar=progress, options=qt.Options(nsteps=nsteps, max_step=max_step)).states
        else:
            pass
            # full_result = qt.mcsolve(self.H_solver_str(seq), psi0, times, c_ops, progress_bar=progress, options=qt.Options(nsteps=nsteps))
            # return np.sum(full_result.states, axis=0)/full_result.ntraj
    
    def evolve_unrotate(self, times, result=None, psi0=None, seq:PulseSequence=None, H=None, c_ops=None, nsteps=1000, max_step=0.1, progress=True):
        # if not progress: progress = None
        # if c_ops is None:
        #     return qt.mesolve(self.H_solver_unrotate(seq=seq, H=H), psi0, times, progress_bar=progress, options=qt.Options(nsteps=nsteps, max_step=max_step)).states
        # else:
        #     full_result = qt.mcsolve(self.H_solver_unrotate(seq), psi0, times, c_ops, progress_bar=progress, options=qt.Options(nsteps=nsteps, max_step=max_step))
        #     return np.sum(full_result.states, axis=0)/full_result.ntraj
        if result is None:
            result = self.evolve(psi0=psi0, times=times, seq=seq, H=H, c_ops=c_ops, nsteps=nsteps, max_step=max_step, progress=progress)
        if H is None:
            H = self.H
            esys = self.esys
        else: esys = H.eigenstates()
        assert len(result) == len(times)
        result_rot = [0*result[i_t] for i_t in range(len(times))]
        for i_t, t in enumerate(tqdm(times, disable=not progress)):
            evals, evecs = esys
            for eval, evec in zip(evals, evecs):
                # |a> = sum_i(|biXbi|a>), a.overlap(b) = <a|b>
                result_rot[i_t] += np.exp(1j*eval*t) * evec.overlap(result[i_t]) * evec
        return result_rot


    # ======================================= #
    # Characterization
    # ======================================= #
    def fidelity(self, ket_target, ket_actual):
        return np.abs(ket_actual.overlap(ket_target))**2

# ======================================================================================= #
# ======================================================================================= #
# ======================================================================================= #

class QSwitchTunableTransmonCoupler(QSwitch):
    def EJ_flux(self, EJmax=None, phi_ext=None, dEJ=None): # phi_ext in units of phi0
        if EJmax is None: EJmax = self.EJs[-1]
        if phi_ext is None: phi_ext = self.phi_ext
        if dEJ is None: dEJ = self.dEJ
        return EJmax * np.sqrt(np.cos(np.pi*phi_ext)**2 + dEJ**2 * np.sin(np.pi*phi_ext)**2)

    """
    Assume coupler is between Q0 and Q1, indexed as last element for all parameter arrays
    """
    def __init__(
        self,
        EJs=None, ECs=None,
        dEJ=0, gs=None, # dEJ=(EJ1-EJ2)/(EJ1+EJ2); gs=[01, 12, 13, 02, 03, 23, 0c, 1c, 2c, 3c] ** MAKE SURE TO SPECIFY gs AT 0 FLUX
        qubit_freqs=None, alphas=None, # specify either frequencies + anharmonicities or qubit parameters
        phi_ext=0, # external flux in units of Phi0 applied to coupler
        useZZs=False, ZZs=None, # specify qubit freqs and ZZ shifts to construct H instead, aka dispersive hamiltonian
        cutoffs=None,
        is2Q=False, # Model 2 coupled transmons with coupler instead of full QRAM module
        isCavity=[False, False, False, False, False]) -> None:

        self.is2Q = is2Q
        self.useZZs = useZZs
        self.nqubits = 2 + 2*(not is2Q) + 1
        self.isCavity = isCavity

        if is2Q and cutoffs is None: cutoffs = [5, 5, 5]
        elif cutoffs is None: cutoffs = [4,5,4,4,5]
        self.cutoffs = cutoffs

        self.alphas = np.array(alphas)
        self.phi_ext = phi_ext

        if self.useZZs:
            assert qubit_freqs is not None and ZZs is not None
            self.qubit_freqs = np.array(qubit_freqs) # w*adag*a = w*sigmaZ/2
            self.ZZs = np.array(ZZs)
        else:
            assert gs is not None
            if np.array(gs).ndim == 0:
                gs = np.array([gs])
            self.gs = gs

            if qubit_freqs is not None and alphas is not None:
                self.qubit_freqs = np.array(qubit_freqs)

            else:
                assert EJs is not None and ECs is not None and gs is not None
                self.EJs = EJs
                self.ECs = ECs
                self.gs = gs
                self.dEJ = dEJ

                # self.qubit_freqs = [self.transmon_fge(ECs[i], EJs[i]) for i in range(self.nqubits - 1)]
                # EJc = self.EJ_flux(EJmax=EJs[-1], phi_ext=phi_ext, dEJ=self.dEJ)
                # self.qubit_freqs.append(self.transmon_fge(ECs[-1], EJc))
                # self.alphas = [(not isCavity[i])*(self.transmon_alpha(ECs[i], EJs[i])) for i in range(self.nqubits - 1)]
                # self.alphas.append(self.transmon_alpha(ECs[-1], EJc))

                transmons = [scq.Transmon(EC=ECs[i], EJ=EJs[i], ng=0, ncut=110, truncated_dim=cutoffs[i]) for i in range(self.nqubits - 1)]
                transmons.append(scq.TunableTransmon(EJmax=self.EJs[-1], EC=self.ECs[-1], d=self.dEJ, flux=phi_ext, ng=0, ncut=110))
    
                evals = [None]*self.nqubits
                evecs = [None]*self.nqubits
                for i in range(self.nqubits):
                    evals[i], evecs[i] = transmons[i].eigensys(evals_count=cutoffs[i])
                    evals[i] -= evals[i][0]
                self.qubit_freqs = [evals[i][1] for i in range(self.nqubits)]
                self.alphas = [(not isCavity[i]) * evals[i][2] - 2*evals[i][1] for i in range(self.nqubits)]

        # create the annihilation ops for each qubit
        self.id_op = qt.tensor(*[qt.qeye(cutoffs[i]) for i in range(self.nqubits)])
        self.a_ops = [None]*self.nqubits
        for q in range(self.nqubits):
            aq = [qt.qeye(cutoffs[i]) if i != q else qt.destroy(cutoffs[i]) for i in range(self.nqubits)]
            aq = qt.tensor(*aq)
            self.a_ops[q] = aq

        # construct qubit hamiltonians
        self.H_no_coupler = 0*self.id_op
        self.H0 = 0*self.id_op
        for q in range(self.nqubits - 1):
            a = self.a_ops[q]
            self.H_no_coupler += 2*np.pi*(self.qubit_freqs[q]*a.dag()*a + 1/2*self.alphas[q]*a.dag()*a.dag()*a*a)
        ac = self.a_ops[-1]
        self.H0 = self.H_no_coupler + 2*np.pi*(self.qubit_freqs[-1]*ac.dag()*ac + 1/2*self.alphas[-1]*ac.dag()*ac.dag()*ac*ac)

        if not self.useZZs:
            # gs=[01, 12, 13, 02, 03, 23, 0c, 1c, 2c, 3c]
            # gs=[01, 0c, 1c]
            a = self.a_ops[0]
            b = self.a_ops[1]
            ac = self.a_ops[-1]
            self.H_int_01 = 2*np.pi*self.gs[0] * (a * b.dag() + a.dag() * b)
            self.H_int_0c = 2*np.pi*self.gs[-self.nqubits + 1] * (a * ac.dag() + a.dag() * ac)
            self.H_int_1c = 2*np.pi*self.gs[-self.nqubits + 2] * (b * ac.dag() + b.dag() * ac)
            self.H_int_no_coupler = self.H_int_01
            self.H_int_coupler = self.H_int_0c + self.H_int_1c
            if not is2Q:
                c = self.a_ops[2]
                d = self.a_ops[3]
                self.H_int_12 = 2*np.pi*self.gs[1] * (b * c.dag() + b.dag() * c)
                self.H_int_13 = 2*np.pi*self.gs[2] * (b * d.dag() + b.dag() * d)
                self.H_int_no_coupler += self.H_int_12 + self.H_int_13

                if len(self.gs) == 10: 
                    self.H_int_02 = 2*np.pi*self.gs[3] * (a * c.dag() + a.dag() * c)
                    self.H_int_03 = 2*np.pi*self.gs[4] * (a * d.dag() + a.dag() * d)
                    self.H_int_23 = 2*np.pi*self.gs[5] * (c * d.dag() + c.dag() * d)
                    self.H_int_no_coupler += self.H_int_02 + self.H_int_03 + self.H_int_23

                    self.H_int_2c = 2*np.pi*self.gs[-2] * (c * ac.dag() + c.dag() * ac)
                    self.H_int_3c = 2*np.pi*self.gs[-1] * (d * ac.dag() + d.dag() * ac)
                    self.H_int_coupler += self.H_int_2c + self.H_int_3c
            # make sure to put in flux dependence of g
            self.H_int = self.H_int_no_coupler + self.H_int_coupler * np.sqrt(self.EJ_flux(EJmax=1, phi_ext=self.phi_ext, dEJ=self.dEJ)) # sqrt here because it goes as impedance

        else: # use ZZ shift values: what is the adjustment to qi when qj is in e?
            assert False, 'not implemented ZZ shift instantiation yet!'
            ZZs = (self.ZZs + np.transpose(self.ZZs))/2 # average over J_ml and J_lm
            self.H_int_no_coupler = 0*self.H_no_coupler
            self.H_int = 0*self.H_no_coupler
            for i in range(len(ZZs)):
                for j in range(i+1, len(ZZs[0])):
                    tensor = [qt.qeye(cutoffs[q]) for q in range(self.nqubits)]
                    tensor[i] = qt.destroy(cutoffs[i]).dag() * qt.destroy(cutoffs[i])
                    tensor[j] = qt.destroy(cutoffs[j]).dag() * qt.destroy(cutoffs[j])
                    self.H_int += 2*np.pi*ZZs[i, j]*qt.tensor(*tensor) # adag_i * a_i * adag_j * a_j

        self.H = self.H0 + self.H_int
        self.esys = self.H.eigenstates()

        # Time independent drive op w/o drive amp.
        # This assumes time dependence given by sin(wt).
        # If given by exp(+/-iwt), need to divide by 2.
        self.flux_drive_ops = [2*np.pi*(a.dag()*a) for a in self.a_ops]

        # Charge drive
        self.drive_ops = [2*np.pi*(a.dag() + a) for a in self.a_ops]

    def get_H_at_phi_ext(self, phi_ext):
        transmon = scq.TunableTransmon(EJmax=self.EJs[-1], EC=self.ECs[-1], d=self.dEJ, flux=phi_ext, ng=0, ncut=110)
        evals, evecs = transmon.eigensys(evals_count=self.cutoffs[-1])
        evals -= evals[0]
        qubit_freq = evals[1]
        alpha = (not self.isCavity[-1]) * evals[2] - 2*evals[1]
        ac = self.a_ops[-1]
        coupler_H0 = 2*np.pi*(qubit_freq*ac.dag()*ac + 1/2*alpha*ac.dag()*ac.dag()*ac*ac)
        H0 = self.H_no_coupler +  coupler_H0

        H_int = self.H_int_no_coupler + self.H_int_coupler * np.sqrt(self.EJ_flux(EJmax=1, phi_ext=self.phi_ext, dEJ=self.dEJ))
        H = H0 + H_int
        return coupler_H0, H0, H
        
    def update_H(self, phi_ext, solve_esys=True):
        coupler_H0, H0, H = self.get_H_at_phi_ext(phi_ext)
        self.H0 = H0
        self.H = H
        if solve_esys: self.esys = self.H.eigenstates()
        else: self.esys = None


    # go from state1 -> state2
    def get_base_wd(self, state1, state2, keep_sign=False, phi_ext=None, esys=None, **kwargs):
        assert not (phi_ext is None and esys is not None)
        if phi_ext is None and esys is None:
            H = self.H
            esys = self.esys
        if phi_ext is not None:
            coupler_H0, H0, H = self.get_H_at_phi_ext(phi_ext)
            if esys is not None: esys = H.eigenstates()

        wd = qt.expect(H, self.state(state2, esys=esys)) - qt.expect(H, self.state(state1, esys=esys))
        if keep_sign: return wd
        return np.abs(wd)

    def state(self, levels, phi_ext=None, esys=None):
        if phi_ext is None and esys is None: esys = self.esys
        if phi_ext is not None:
            coupler_H0, H0, H = self.get_H_at_phi_ext(phi_ext)
            esys = H.eigenstates()
        return self.find_dressed(self.make_bare(levels), esys=esys)[2]

    # ======================================= #
    # Modified sequence constructor for flux drive: goal is to apply (wc(t) - wc0)a^dag*a
    # ======================================= #
    def flux_drive_modulation_wc(self, dc_phi_ext):
        assert self.phi_ext == 0, "This function assumes the qubit frequencies were evaluated at phi=0"
        def modulation(t, wd, amp, phase):
            return self.qubit_freqs[-1]*(np.sqrt(np.abs(np.cos(np.pi*(dc_phi_ext + amp*np.cos(wd*t + phase))))) - 1)
        return modulation

    def flux_drive_modulation_H_int_coupler(self, dc_phi_ext):
        assert self.phi_ext == 0, "This function assumes the qubit frequencies were evaluated at phi=0"
        def modulation(t, wd, amp, phase):
            phi_ext_osc = dc_phi_ext + amp*np.cos(wd*t + phase)
            return np.sqrt(self.EJ_flux(EJmax=1, phi_ext=phi_ext_osc, dEJ=self.dEJ)) - 1
        return modulation

    def flux_drive_sequence(self, dc_phi_ext, seq, seq_func, **kwargs):
        # print(dc_phi_ext)
        kwargs['modulation'] = self.flux_drive_modulation_wc(dc_phi_ext)
        t_start = seq.time + (kwargs['t_offset'] if 't_offset' in kwargs else 0)
        seq_func(**kwargs)
        kwargs['modulation'] = self.flux_drive_modulation_H_int_coupler(dc_phi_ext)
        kwargs['t_start'] = t_start
        seq_func(**kwargs)

    def H_solver_flux_drive(self, seq:PulseSequence, H=None):
        if H is None: H = self.H
        H_solver = [H]
        pulse_seq = seq.get_pulse_seq()
        for pulse_i in range(0, len(pulse_seq), 2):
            # print('pulse_i', pulse_i)
            pulse_func_wc = pulse_seq[pulse_i]
            pulse_func_H_int_coupler = pulse_seq[pulse_i + 1]
            H_solver.append([self.flux_drive_ops[seq.drive_qubits[pulse_i]], pulse_func_wc])
            H_solver.append([self.H_int_coupler, pulse_func_H_int_coupler])
        return H_solver


    """
    flux_seq should be an array of same length as number of pulses in sequence that indicates what phi_ext is at each step
    """
    def H_flux_sequence(self, flux_seq, H=None):
        H_seq = []
        for phi_ext in flux_seq:
            coupler_H0, H0, H = self.get_H_at_phi_ext(phi_ext)
            H_seq.append(H)
        return H_seq


    def H_solver_flux_sequence(self, seq:PulseSequence, sigma_n=4):
        window_funcs, flux_seq = seq.construct_flux_transition_map(sigma_n=sigma_n)
        H_seq = self.H_flux_sequence(flux_seq)

        H_solver = []
        for flux_i in range(len(flux_seq)):
            H_solver.append([H_seq[flux_i], window_funcs[flux_i]])

        pulse_seq = seq.get_pulse_seq()
        for seq_i in range(len(pulse_seq)):
            H_solver.append([self.drive_ops[seq.drive_qubits[seq_i]], pulse_seq[seq_i]])

        return H_solver


    def evolve_flux_sequence(self, psi0, times=None, seq:PulseSequence=None, flux_seq=None, flux_transit_time=6, progress=True, nsteps=10000, H_solver=None):
        if H_solver is None:
            H_solver = self.H_solver_flux_sequence(seq=seq, flux_seq=flux_seq, times=times, flux_transit_time=flux_transit_time)
        if not progress: progress = None
        return qt.mesolve(H_solver, psi0, times, progress_bar=progress, options=qt.Options(nsteps=nsteps)).states




# ======================================================================================= #
# ======================================================================================= #
# ======================================================================================= #

class QSwitchSNAIL(QSwitch):
    """
    Assume coupler is between Q0 and Q1, indexed as last element for all parameter arrays
    """

    def snail_potential(self, phi_vec, phi_ext=0):
        '''
        Return the potential energy of the SNAIL qubit
        N = number of small junctions
        beta = ratio of big junction to small junctions
        phi_vec=vector of phase values to evalute over
        phi_ext=2pi*flux/Phi0
        '''
        EJ = self.EJs[-1]
        EL = self.EL
        N = self.N
        beta = self.beta

        U_eff = -beta*EJ*np.cos(phi_vec - phi_ext) - N*EJ*np.cos(phi_vec/N)

        return U_eff

    def snail_potential_coeff(self, phi_ext=0):

        '''
        Return Taylor expansion coefficients of the potential energy of the SNAIL qubit
        Provide phi_ext in units of Phi0
        '''
        EJ = self.EJs[-1]
        EC = self.ECs[-1]
        EL = self.EL
        N = self.N
        beta = self.beta

        # Find the minimum of the potential
        phi_vec = np.linspace(0, 2*np.pi*N, 100000)
        U_vec = self.snail_potential(phi_vec, 2*np.pi*phi_ext)
        phi_min = phi_vec[np.argmin(U_vec)]

        c2 = beta*np.cos(phi_min - 2*np.pi*phi_ext) + 1/N*np.cos(phi_min/N)
        c3 = -(N**2 - 1)/N**2*np.sin(phi_min/N)
        # c3 = -beta*np.sin(phi_min - phi_ext) - 1/(N**2)*np.sin(phi_min/N)
        c4 = - beta*np.cos(phi_min - 2*np.pi*phi_ext) - 1/N**3*np.cos(phi_min/N)
        c5 = -(1 - N**4)/N**4*np.sin(phi_min/N)

        p = 0
        if EL != None:

            p = EL / (EJ*c2 + EL)

            c2r = p*c2
            c3r = p**3*c3
            c4r = p**4*(c4 - 3*c3**2/c2*(1 - p))
            c5r = p**5*(c5 - 10*c4*c3/c2*(1 - p) + 15*c3**2/c2**2*(1 - p)**2)
            
            # renormalized by EL
            c2 = c2r
            c3 = c3r
            c4 = c4r
            c5 = c5r
        # print(phi_ext, c2, c3, c4, c5, p, phi_min)
        w_sn = np.sqrt(8*EC * EJ * c2)

        g3 = 1/6 * p**2/N * c3/c2 * np.sqrt(EC*w_sn/2/np.pi)
        g4 = 1/12 * p**3/N**2  * (c4 - 3*c3**2/c2 * (1-p)) * EC/c2

        return c2, c3, c4, c5, p, g3, g4, phi_min


    def __init__(
        self,
        EJs=None, ECs=None, gs=None, # gs=[01, 12, 13, 02, 03, 23, 0c, 1c, 2c, 3c]
        qubit_freqs=None, alphas=None, # specify either frequencies + anharmonicities or qubit parameters
        beta=None, N=None, EL=None, phi_ext=0, # external flux in units of Phi0 applied to coupler
        useZZs=False, ZZs=None, # specify qubit freqs and ZZ shifts to construct H instead, aka dispersive hamiltonian
        cutoffs=None,
        is2Q=False, # Model 2 coupled transmons with coupler instead of full QRAM module
        isCavity=[False, False, False, False, False],
        solve_esys=True) -> None:

        self.is2Q = is2Q
        self.useZZs = useZZs
        self.nqubits = 2 + 2*(not is2Q) + 1

        if is2Q and cutoffs is None: cutoffs = [5, 5, 5]
        elif cutoffs is None: cutoffs = [4,5,4,4,5]
        self.cutoffs = cutoffs

        self.alphas = np.array(alphas)

        if self.useZZs:
            assert qubit_freqs is not None and ZZs is not None
            self.qubit_freqs = np.array(qubit_freqs) # w*adag*a = w*sigmaZ/2
            self.ZZs = np.array(ZZs)
        else:
            assert gs is not None
            if np.array(gs).ndim == 0:
                gs = np.array([gs])
            self.gs = gs

            if qubit_freqs is not None and alphas is not None:
                self.qubit_freqs = np.array(qubit_freqs)

            else:
                assert EJs is not None and ECs is not None and gs is not None
                self.EJs = EJs
                self.ECs = ECs
                self.gs = gs
                self.beta = beta
                self.N = N
                self.EL = EL
                transmons = [scq.Transmon(EC=ECs[i], EJ=EJs[i], ng=0, ncut=110, truncated_dim=cutoffs[i]) for i in range(self.nqubits - 1)]
    
                evals = [None]*self.nqubits
                evecs = [None]*self.nqubits
                for i in range(self.nqubits - 1):
                    evals[i], evecs[i] = transmons[i].eigensys(evals_count=cutoffs[i])
                    evals[i] -= evals[i][0]
                self.qubit_freqs = [evals[i][1] for i in range(self.nqubits - 1)]
                c2, c3, c4, c5, p, g3, g4, phi_min = self.snail_potential_coeff(phi_ext=phi_ext)
                self.qubit_freqs.append(np.sqrt(8*ECs[-1] * EJs[-1] * c2))
                self.alphas = [(not isCavity[i]) * evals[i][2] - 2*evals[i][1] for i in range(self.nqubits - 1)]
                self.alphas.append(0)

        self.id_op = qt.tensor(*[qt.qeye(cutoffs[i]) for i in range(self.nqubits)])
        self.a_ops = [None]*self.nqubits
        for q in range(self.nqubits):
            aq = [qt.qeye(cutoffs[i]) if i != q else qt.destroy(cutoffs[i]) for i in range(self.nqubits)]
            aq = qt.tensor(*aq)
            self.a_ops[q] = aq

        self.H0 = 0*self.id_op
        for q in range(self.nqubits):
            a = self.a_ops[q]
            self.H0 += 2*np.pi*(self.qubit_freqs[q]*a.dag()*a + 1/2*self.alphas[q]*a.dag()*a.dag()*a*a)
            if q == self.nqubits - 1:
                wsn = self.qubit_freqs[-1]
                self.H0 += 2*np.pi*(g3*(a+a.dag())**3 + g4*(a+a.dag())**4)

        if not self.useZZs:
            # gs=[01, 12, 13, 02, 03, 23, 0c, 1c, 2c, 3c]
            # gs=[01, 0c, 1c]
            a = self.a_ops[0]
            b = self.a_ops[1]
            ac = self.a_ops[-1]
            self.H_int_01 = 2*np.pi*self.gs[0] * (a * b.dag() + a.dag() * b)
            self.H_int_0c = 2*np.pi*self.gs[-self.nqubits + 1] * (a * ac.dag() + a.dag() * ac)
            self.H_int_1c = 2*np.pi*self.gs[-self.nqubits + 2] * (b * ac.dag() + b.dag() * ac)
            self.H_int = self.H_int_01 + self.H_int_0c + self.H_int_1c
            if not is2Q:
                c = self.a_ops[2]
                d = self.a_ops[3]
                self.H_int_12 = 2*np.pi*self.gs[1] * (b * c.dag() + b.dag() * c)
                self.H_int_13 = 2*np.pi*self.gs[2] * (b * d.dag() + b.dag() * d)
                self.H_int += self.H_int_12 + self.H_int_13
                if len(self.gs) == 6: 
                    self.H_int_02 = 2*np.pi*self.gs[3] * (a * c.dag() + a.dag() * c)
                    self.H_int_03 = 2*np.pi*self.gs[4] * (a * d.dag() + a.dag() * d)
                    self.H_int_23 = 2*np.pi*self.gs[5] * (c * d.dag() + c.dag() * d)
                    self.H_int_2c = 2*np.pi*self.gs[-2] * (c * ac.dag() + c.dag() * ac)
                    self.H_int_3c = 2*np.pi*self.gs[-1] * (d * ac.dag() + d.dag() * ac)
                    self.H_int += self.H_int_02 + self.H_int_03 + self.H_int_23 + self.H_int2c + self.H_int3c
        else: # use ZZ shift values: what is the adjustment to qi when qj is in e?
            assert False, 'not implemented ZZ shift instantiation yet!'
            ZZs = (self.ZZs + np.transpose(self.ZZs))/2 # average over J_ml and J_lm
            self.H_int = 0*self.H0
            for i in range(len(ZZs)):
                for j in range(i+1, len(ZZs[0])):
                    tensor = [qt.qeye(cutoffs[q]) for q in range(self.nqubits)]
                    tensor[i] = qt.destroy(cutoffs[i]).dag() * qt.destroy(cutoffs[i])
                    tensor[j] = qt.destroy(cutoffs[j]).dag() * qt.destroy(cutoffs[j])
                    self.H_int += 2*np.pi*ZZs[i, j]*qt.tensor(*tensor) # adag_i * a_i * adag_j * a_j
        self.H = self.H0 + self.H_int
        if solve_esys: self.esys = self.H.eigenstates()

        # Time independent drive op w/o drive amp.
        # This assumes time dependence given by sin(wt).
        # If given by exp(+/-iwt), need to divide by 2.
        self.drive_ops = [2*np.pi*(a.dag() + a) for a in self.a_ops]