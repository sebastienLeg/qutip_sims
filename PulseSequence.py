from pyclbr import Function
import numpy as np
import qutip as qt
from typing import List, Set, Dict, Tuple
import scipy as sp

# Gaussian with amp = 1
def gaussian(x, sigma):
    return np.exp(-x**2/2/sigma**2)

def logistic(x, sigma):
    return 1 / (1 + np.exp(-x/sigma))

# ====================================================== #
# Adiabatic pi pulse functions
# beta ~ slope of the frequency sweep (also adjusts width)
# mu ~ width of frequency sweep (also adjusts slope)
# period: delta frequency sweeps through zero at period/2
# amp_max

def adiabatic_amp(t, amp_max, beta, period):
    return amp_max / np.cosh(beta*(2*t/period - 1))

def adiabatic_phase(t, mu, beta, period):
    return mu * np.log(adiabatic_amp(t, amp_max=1, beta=beta, period=period))

def adiabatic_iqamp(t, amp_max, mu, beta, period):
    amp = np.abs(adiabatic_amp(t, amp_max=amp_max, beta=beta, period=period))
    phase = adiabatic_phase(t, mu=mu, beta=beta, period=period)
    iamp = amp * (np.cos(phase) + 1j*np.sin(phase))
    qamp = amp * (-np.sin(phase) + 1j*np.cos(phase))
    return np.real(iamp), np.real(qamp)

# ====================================================== #

class PulseSequence:
    def __init__(self, start_time=0):
        self.pulse_seq = []
        self.envelope_seq = [] # normalized envelopes
        self.drive_qubits = []
        self.pulse_lengths = []
        self.pulse_freqs = [] # pulse frequencies (real freq, not omega)
        self.pulse_phases = [] # pulse phases (radians)
        self.pulse_strs = [] # cython strs for pulses
        self.time = start_time
        self.start_times = []
        self.pulse_names = [] # list of tuples, every tuple is the levels b/w which the pulse operates, alphabetically listed
        self.amps = [] # list of amplitudes (real freq, not omega)
        self.flux_transitions = [] # list of flux transitions in format (t_begin_transit, t_end_transit, flux_f)

    def get_pulse_seq(self):
        return self.pulse_seq

    def get_envelope_seq(self):
        return self.envelope_seq

    def get_pulse_names(self, simplified=False):
        if not simplified: return self.pulse_names
        else: return list(set(self.pulse_names))

    def get_drive_qubits(self, simplified=False):
        if not simplified: return self.drive_qubits
        else:
            pulse_to_drive_qubits = dict()
            for i, pulse in enumerate(self.pulse_names):
                if pulse in pulse_to_drive_qubits: continue
                pulse_to_drive_qubits.update({pulse:self.drive_qubits[i]})
            return pulse_to_drive_qubits

    def get_pulse_lengths(self):
        return np.array(self.pulse_lengths)

    def get_start_times(self):
        return np.array(self.start_times)

    def get_pulse_freqs(self, simplified=False):
        if not simplified: return self.pulse_freqs
        else:
            pulse_to_freqs = dict()
            for i, pulse in enumerate(self.pulse_names):
                if pulse in pulse_to_freqs: continue
                pulse_to_freqs.update({pulse:self.pulse_freqs[i]})
            return pulse_to_freqs

    def get_pulse_phases(self):
        return np.array(self.pulse_phases)

    def get_pulse_amps(self, simplified=False):
        if not simplified: return np.array(self.amps)
        else:
            pulse_to_amps = dict()
            for i, pulse in enumerate(self.pulse_names):
                if pulse in pulse_to_amps: continue
                pulse_to_amps.update({pulse:self.amps[i]})
            return np.array(pulse_to_amps)

    def get_pulse_str(self):
        pulse_str_drive_qubit = ['0']*4
        for pulse_i, pulse_str in enumerate(self.pulse_strs):
            pulse_str_drive_qubit[self.drive_qubits[pulse_i]] += '+' + pulse_str
        return pulse_str_drive_qubit

    def get_flux_transitions(self):
        return np.array(self.flux_transitions)
    
    def get_end_time(self): # end time of full pulse sequence, including empty wait times
        end_pulse_times_q = dict()
        for q, pulse_len, start_time in zip(self.get_drive_qubits(), self.get_pulse_lengths(), self.get_start_times()):
            if str(q) not in end_pulse_times_q.keys():
                end_pulse_times_q.update({str(q):start_time + pulse_len})
            else: end_pulse_times_q.update({str(q):max((end_pulse_times_q[str(q)], start_time + pulse_len))})

        end_pulse_times_q_arr = []
        for q in end_pulse_times_q.keys():
            end_pulse_times_q_arr.append(end_pulse_times_q[q])

        end_time = 0
        if len(end_pulse_times_q_arr) > 0:
            end_time = max(end_pulse_times_q_arr)
        return end_time
            




    """
    Advance current time by t (marker indicating end of last pulse)
    This is automatically done when calling pulse functions
    """
    def wait(self, t):
        self.time += t

    def prev_pulse_length(self):
        return self.pulse_lengths[-1]

    def pulse(self, t, args):
        return sum([pulse_i(t, args) for pulse_i in self.pulse_seq])
    
    """
    Adds the drive_func corresponding to a constant pulse with a sin^2
    ramp up/down to the sequence.
    t_offset is offset from the end of the last pulse.
    t_start, if not None, defines the absolute pulse start time relative to the beginning of the pulse sequence (overrides t_offset)
    amp: freq
    phase: radians
    Returns the total length of the sequence.
    """
    def const_pulse(self, wd=None, amp=None, t_pulse=None, pulse_levels:Tuple[str,str]=None, modulation:Function=None, drive_qubit=1, t_offset=0, t_start=None, t_rise=1, phase=0):
        assert None not in [t_pulse, drive_qubit]
        if t_start is None: t_start = self.time + t_offset
        self.time = t_start + t_pulse

        envelope = None
        drive_func = None
        c_str = None
        if None in pulse_levels: pulse_levels = None

        if amp is not None:
            assert None not in [wd, pulse_levels]
            def envelope(t, args=None):
                    t -= t_start 
                    if 0 <= t < t_rise: return np.sin(np.pi*t/2/t_rise)**2
                    elif t_rise <= t < t_pulse - t_rise: return 1
                    elif t_pulse - t_rise <= t < t_pulse: return np.sin(np.pi*(t_pulse-t)/2/t_rise)**2
                    else: return 0 
            def drive_func(t, args=None):
                if modulation is None: return amp*envelope(t)*np.cos(wd*t + phase)
                else: return envelope(t)*modulation(t, wd, amp, phase)

            c_str = f'({amp}) * cos(({wd})*t+{phase}) * ('
            c_str += f'sin(pi*(t-({t_start}))/2/({t_rise}))*sin(pi*(t-({t_start}))/2/({t_rise})) * (np.heaviside(t-({t_start}),0)-np.heaviside(t-({t_start})-({t_rise}),0))'
            c_str += f' + (np.heaviside(t-({t_start})-({t_rise}),0)-np.heaviside(t-({t_start})-({t_pulse})-({t_rise}),0))'
            c_str += f' + sin(pi*(t-({t_start}))/2/({t_rise}))*sin(pi*(t-({t_start}))/2/({t_rise})) * (np.heaviside(t-({t_start})-({t_pulse})-({t_rise}),0)-np.heaviside(t-({t_start})-({t_pulse}),0))'
            c_str += ')'

        self.start_times.append(t_start)
        self.pulse_strs.append(c_str)
        self.envelope_seq.append(envelope)
        self.drive_qubits.append(drive_qubit)
        self.pulse_seq.append(drive_func)
        self.pulse_lengths.append(t_pulse)
        self.pulse_freqs.append(wd/2/np.pi if wd is not None else None)
        self.pulse_phases.append(phase)
        self.pulse_names.append((min(pulse_levels), max(pulse_levels)) if pulse_levels is not None else None)
        self.amps.append(amp)

    """
    Adds the drive_func corresponding to a gaussian pulse to the sequence.
    Returns the total length of the sequence.
    """
    def gaussian_pulse(self, wd=None, amp=None, t_pulse_sigma=None, pulse_levels:Tuple[str,str]=None, sigma_n=4, modulation:Function=None, drive_qubit=1, t_offset=0, t_start=None, phase=0):
        assert None not in [t_pulse_sigma, drive_qubit]
        if t_start is None: t_start = self.time + t_offset
        self.time = t_start + sigma_n*t_pulse_sigma

        envelope = None
        drive_func = None
        if None in pulse_levels: pulse_levels = None

        if amp is not None:
            assert None not in [wd, pulse_levels]
            def envelope(t, args=None):
                t_max = t_start + sigma_n/2*t_pulse_sigma # point of max in gaussian
                if t < t_start or t > t_start + sigma_n*t_pulse_sigma: return 0
                return gaussian(t - t_max, t_pulse_sigma)
            def drive_func(t, args=None):
                if modulation is None:
                    # print(amp, t_start, envelope(t))
                    return amp*envelope(t)*np.cos(wd*t + phase)
                else: return envelope(t)*modulation(t, wd, amp, phase)
                # return amp*envelope(t)*np.sin(wd*t - phase)

        self.start_times.append(t_start)
        self.envelope_seq.append(envelope)
        self.pulse_seq.append(drive_func)
        self.drive_qubits.append(drive_qubit)
        self.pulse_lengths.append(sigma_n*t_pulse_sigma)
        self.pulse_freqs.append(wd/2/np.pi if wd is not None else None)
        self.pulse_phases.append(phase)
        self.pulse_names.append((min(pulse_levels), max(pulse_levels)) if pulse_levels is not None else None)
        self.amps.append(amp)

    def adiabatic_pulse(self, wd=None, amp=None, mu=None, beta=None, period=None, pulse_levels:Tuple[str,str]=None, drive_qubit=1, t_offset=0, t_start=None, phase=0):
        assert None not in [period, drive_qubit]
        if t_start is None: t_start = self.time + t_offset
        self.time = t_start + period

        envelope = None
        drive_func = None
        if None in pulse_levels: pulse_levels = None

        if amp is not None:
            assert None not in [wd, mu, beta, pulse_levels]
            def envelope(t, args=None):
                if t < t_start or t > t_start + period: return 0
                return adiabatic_amp(t-t_start, amp_max=1, beta=beta, period=period)
            def drive_func(t, args=None):
                phase_t = adiabatic_phase(t-t_start, mu=mu, beta=beta, period=period)
                return amp*envelope(t)*(np.cos(phase_t)*np.cos(wd*t + phase) - np.sin(phase_t)*np.sin(wd*t + phase))

        self.start_times.append(t_start)
        self.envelope_seq.append(envelope)
        self.pulse_seq.append(drive_func)
        self.drive_qubits.append(drive_qubit)
        self.pulse_lengths.append(period)
        self.pulse_freqs.append(wd/2/np.pi if wd is not None else None)
        self.pulse_phases.append(phase)
        self.pulse_names.append((min(pulse_levels), max(pulse_levels)) if pulse_levels is not None else None)
        self.amps.append(amp)

    """
    Flat top pulse with gaussian ramp up/down
    t_offset is offset from the end of the last pulse.
    t_start, if not None, defines the absolute pulse start time relative to the beginning of the pulse sequence (overrides t_offset)
    amp: freq
    phase: radians
    t_pulse: includes total length of pulse
    t_rise: length of ramp up = length of ramp down
    sigma_n: number of sigmas for ramp up = number of sigmas for ramp down
    Returns the total length of the sequence.
    """
    def flat_top_pulse(self, wd=None, amp=None, t_pulse=None, pulse_levels:Tuple[str,str]=None, envelope:Function=None, modulation:Function=None, drive_qubit=1, t_offset=0, t_start=None, t_rise=15, sigma_n=2, phase=0):
        assert None not in [t_pulse, drive_qubit]
        if t_start is None: t_start = self.time + t_offset
        self.time = t_start + t_pulse

        envelope = None
        drive_func = None
        if None in pulse_levels: pulse_levels = None

        if amp is not None:
            assert None not in [wd, pulse_levels]
            if envelope is None:
                def envelope(t, args=None):
                        t -= t_start 
                        t_ramp_sigma = t_rise/sigma_n
                        if 0 <= t < t_rise: return gaussian(t-t_rise, t_ramp_sigma)
                        elif t_rise <= t < t_pulse - t_rise: return 1
                        elif t_pulse - t_rise <= t < t_pulse: return gaussian(t-(t_pulse-t_rise), t_ramp_sigma)
                        else: return 0 
            def drive_func(t, args=None):
                if modulation is None: return amp*envelope(t)*np.cos(wd*t + phase)
                else: return envelope(t)*modulation(t, wd, amp, phase)

        self.start_times.append(t_start)
        self.pulse_strs.append(None)
        self.envelope_seq.append(envelope)
        self.drive_qubits.append(drive_qubit)
        self.pulse_seq.append(drive_func)
        self.pulse_lengths.append(t_pulse)
        self.pulse_freqs.append(wd/2/np.pi if wd is not None else None)
        self.pulse_phases.append(phase)
        self.pulse_names.append((min(pulse_levels), max(pulse_levels)) if pulse_levels is not None else None)
        self.amps.append(amp)

    """
    Pulse with I(t)sin(wd t) + Q(t)cos(wd t)
    I_values, Q_values should be arrays of I, Q values evaluated at times
    """
    def pulse_IQ(self, wd=None, amp=None, pulse_levels:Tuple[str,str]=None, I_values=None, Q_values=None, times=None, drive_qubit=1, t_offset=0, t_start=None, phase=0):
        assert None not in [times, drive_qubit]
        if t_start is None: t_start = self.time + t_offset
        self.time = t_start + times[-1]

        envelope = None
        drive_func = None
        if None in pulse_levels: pulse_levels = None

        if amp is not None:
            assert None not in [wd, pulse_levels, I_values, Q_values]
            I_func = sp.interpolate.interp1d(times, I_values, fill_value='extrapolate', kind='quadratic')
            Q_func = sp.interpolate.interp1d(times, Q_values, fill_value='extrapolate', kind='quadratic')
            envelope = [lambda t, args=None: I_func(t), lambda t, args=None: Q_func(t)]
            def drive_func(t, args=None):
                return amp*I_func(t)*np.cos(wd*t + phase) + amp*Q_func(t)*np.sin(wd*t + phase)

        self.start_times.append(t_start)
        self.pulse_strs.append(None)
        self.envelope_seq.append(envelope)
        self.drive_qubits.append(drive_qubit)
        self.pulse_seq.append(drive_func)
        self.pulse_lengths.append(times[-1])
        self.pulse_freqs.append(wd/2/np.pi if wd is not None else None)
        self.pulse_phases.append(phase)
        self.pulse_names.append((min(pulse_levels), max(pulse_levels)) if pulse_levels is not None else None)
        self.amps.append(amp)

    """
    Pulse with 1/2(i I(t) + Q(t))a^dag e^(-i wd t) + h.c.
    Saves just the envelope for the a^dag piece. Need to solve the time
    evolution with H_solver_unrotate to get both components.
    I_values, Q_values should be arrays of I, Q values evaluated at times
    """
    def pulse_IQ_exp(self, wd=None, amp=None, pulse_levels:Tuple[str,str]=None, I_values=None, Q_values=None, times=None, drive_qubit=1, t_offset=0, t_start=None, phase=0):
        assert None not in [times, drive_qubit]
        if t_start is None: t_start = self.time + t_offset
        self.time = t_start + times[-1]

        envelope = None
        drive_func = None
        if None in pulse_levels: pulse_levels = None

        if amp is not None:
            assert None not in [wd, pulse_levels, I_values, Q_values]
            if t_start is None: t_start = self.time + t_offset
            I_func = sp.interpolate.interp1d(times, I_values, fill_value='extrapolate', kind='quadratic')
            Q_func = sp.interpolate.interp1d(times, Q_values, fill_value='extrapolate', kind='quadratic')
            envelope = [lambda t, args=None: I_func(t), lambda t, args=None: Q_func(t)]
            # def drive_func(t, args=None):
            #     return 1/2*amp*(1j*I_func(t) + Q_func(t))*np.exp(-1j*wd*t - phase)
            def drive_func(t, args=None):
                return amp*I_func(t)*np.sin(wd*t - phase) + amp*Q_func(t)*np.cos(wd*t - phase)

        self.start_times.append(t_start)
        self.pulse_strs.append(None)
        self.envelope_seq.append(envelope)
        self.drive_qubits.append(drive_qubit)
        self.pulse_seq.append(drive_func)
        self.pulse_lengths.append(times[-1])
        self.pulse_freqs.append(wd/2/np.pi if wd is not None else None)
        self.pulse_phases.append(phase)
        self.pulse_names.append((min(pulse_levels), max(pulse_levels)) if pulse_levels is not None else None)
        self.amps.append(amp)


    """
    ALWAYS ADD FLUX TRANSITIONS IN ORDER OF START TIME OR ELSE STRANGE THINGS COULD HAPPEN!
    ALSO ASSUMES THAT FLUX TRANSITIONS ARE FAR ENOUGH APART 
    Adds to list of flux transitions in format (t_begin_transit, t_end_transit, flux_f)
    """
    def flux_transition(self, t_flux_rise, flux_f, t_begin_transit=None):
        if t_begin_transit is None: t_begin_transit = self.time
        self.flux_transitions.append([t_begin_transit, t_begin_transit + t_flux_rise, flux_f])
        self.time = t_begin_transit + t_flux_rise
    

    def construct_flux_transition_map(self, sigma_n=6):
        flux_transitions = self.get_flux_transitions()
        assert len(flux_transitions) >= 1 # need at least the "transition" to the starting flux
        flux_seq = flux_transitions[:, 2] # get the flux values we are transitioning to
        transit_start_times = flux_transitions[:, 0]
        transit_end_times = flux_transitions[:, 1]

        t_upramp_start = transit_start_times[0] # time of beginning ramp to current flux
        t_upramp_end = transit_end_times[0] # time of ending ramp to current flux
        t_downramp_start = transit_start_times[1] if len(flux_seq) >= 2 else np.inf # time of beginning ramp to next flux
        t_downramp_end = transit_end_times[1] if len(flux_seq) >= 2 else np.inf # time of ending ramp to next flux

        windows = []
        for flux_i in range(len(flux_seq)):
            def H0_flux_window(
                t_to_i=t_upramp_start, t_to_f=t_upramp_end, 
                t_from_i=t_downramp_start, t_from_f=t_downramp_end,
                args=None):
                # need to define tstart and tend as arguments here or else they will be evaluated only at runtime
                def envelope(t, args=None):
                    if t_to_i <= t < t_to_f:
                        t_ramp_sigma = (t_to_f - t_to_i)/sigma_n/2
                        return logistic(t-sigma_n*t_ramp_sigma-t_to_i, t_ramp_sigma)
                    elif t_to_f <= t < t_from_i:
                        return 1
                    elif t_from_i <= t < t_from_f:
                        t_ramp_sigma = (t_from_f - t_from_i)/sigma_n/2
                        return 1 - logistic(t-sigma_n*t_ramp_sigma-t_from_i, t_ramp_sigma)
                    else: return 0
                return envelope
            windows.append(H0_flux_window())

            if flux_i < len(flux_seq) - 1:
                t_upramp_start = transit_start_times[flux_i + 1]
                t_upramp_end = transit_end_times[flux_i + 1]
                t_downramp_start = transit_start_times[flux_i + 2] if flux_i < len(flux_seq) - 2 else np.inf
                t_downramp_end = transit_end_times[flux_i + 2] if flux_i < len(flux_seq) - 2 else np.inf

        return windows, flux_seq

        