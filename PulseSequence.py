import numpy as np
import qutip as qt
from typing import List, Set, Dict, Tuple

# Gaussian with amp = 1
def gaussian(x, sigma):
    return np.exp(-x**2/2/sigma**2)


class PulseSequence:
    def __init__(self, start_time=0):
        self.pulse_seq = []
        self.envelope_seq = []
        self.drive_qubits = []
        self.pulse_lengths = []
        self.pulse_freqs = [] # pulse frequencies (real freq, not omega)
        self.pulse_strs = [] # cython strs for pulses
        self.time = start_time
        self.pulse_names = [] # list of tuples, every tuple is the levels b/w which the pulse operates, alphabetically listed
        self.amps = [] # list of amplitudes (real freq, not omega)

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
        return self.pulse_lengths

    def get_pulse_freqs(self, simplified=False):
        if not simplified: return self.pulse_freqs
        else:
            pulse_to_freqs = dict()
            for i, pulse in enumerate(self.pulse_names):
                if pulse in pulse_to_freqs: continue
                pulse_to_freqs.update({pulse:self.pulse_freqs[i]})
            return pulse_to_freqs

    def get_pulse_amps(self, simplified=False):
        if not simplified: return self.amps
        else:
            pulse_to_amps = dict()
            for i, pulse in enumerate(self.pulse_names):
                if pulse in pulse_to_amps: continue
                pulse_to_amps.update({pulse:self.amps[i]})
            return pulse_to_amps

    def get_pulse_str(self):
        pulse_str_drive_qubit = ['0']*4
        for pulse_i, pulse_str in enumerate(self.pulse_strs):
            pulse_str_drive_qubit[self.drive_qubits[pulse_i]] += '+' + pulse_str
        return pulse_str_drive_qubit


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
    t_start is offset from the end of the last pulse.
    amp: freq
    Returns the total length of the sequence.
    """
    def const_pulse(self, wd, amp, t_pulse, pulse_levels:Tuple[str,str], drive_qubit=1, t_start=0, t_rise=1):
        t_start = self.time - t_start
        def envelope(t):
                t -= t_start 
                if 0 <= t < t_rise: return amp * np.sin(np.pi*t/2/t_rise)**2
                elif t_rise <= t < t_pulse - t_rise: return amp
                elif t_pulse - t_rise <= t < t_pulse: return amp*np.sin(np.pi*(t_pulse-t)/2/t_rise)**2
                else: return 0 
        def drive_func(t, args):
            return envelope(t)*np.sin(wd*t)

        c_str = f'({amp}) * sin(({wd})*t) * ('
        c_str += f'sin(pi*(t-({t_start}))/2/({t_rise}))*sin(pi*(t-({t_start}))/2/({t_rise})) * (np.heaviside(t-({t_start}),0)-np.heaviside(t-({t_start})-({t_rise}),0))'
        c_str += f' + (np.heaviside(t-({t_start})-({t_rise}),0)-np.heaviside(t-({t_start})-({t_pulse})-({t_rise}),0))'
        c_str += f' + sin(pi*(t-({t_start}))/2/({t_rise}))*sin(pi*(t-({t_start}))/2/({t_rise})) * (np.heaviside(t-({t_start})-({t_pulse})-({t_rise}),0)-np.heaviside(t-({t_start})-({t_pulse}),0))'
        c_str += ')'

        self.pulse_strs.append(c_str)
        self.envelope_seq.append(envelope)
        self.drive_qubits.append(drive_qubit)
        self.pulse_seq.append(drive_func)
        self.pulse_lengths.append(t_pulse)
        self.pulse_freqs.append(wd/2/np.pi)
        self.time = t_start + t_pulse
        self.pulse_names.append((min(pulse_levels), max(pulse_levels)))
        self.amps.append(amp)

    """
    Adds the drive_func corresponding to a gaussian pulse to the sequence.
    Returns the total length of the sequence.
    """
    def gaussian_pulse(self, wd, amp, t_pulse_sigma, pulse_levels:Tuple[str,str], drive_qubit=1, t_start=0):
        t_start = self.time - t_start
        def envelope(t):
                t_max = t_start + 3*t_pulse_sigma
                return amp*gaussian(t - t_max, t_pulse_sigma)
        def drive_func(t, args):
            return envelope(t)*np.sin(wd*t)
        self.envelope_seq.append(envelope)
        self.pulse_seq.append(drive_func)
        self.drive_qubits.apppend(drive_qubit)
        self.pulse_lengths.append(6*t_pulse_sigma)
        self.pulse_freqs.append(wd/2/np.pi)
        self.time = t_start + 6*t_pulse_sigma
        self.pulse_names.append((min(pulse_levels), max(pulse_levels)))
        self.amps.append(amp)