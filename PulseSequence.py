import numpy as np
import qutip as qt

# Gaussian with amp = 1
def gaussian(x, sigma):
    return np.exp(-x**2/2/sigma**2)


class PulseSequence:
    def __init__(self, start_time=0):
        self.pulse_seq = []
        self.envelope_seq = []
        self.pulse_lengths = []
        self.time = start_time


    def get_pulse_seq(self):
        return self.pulse_seq

    def get_envelope_seq(self):
        return self.envelope_seq

    def get_seq_length(self):
        return sum(self.pulse_lengths)
    
    """
    Advance current time by t (marker indicating end of last pulse)
    This is automatically done when calling pulse functions
    """
    def wait(self, t):
        self.time += t

    def prev_pulse_length(self):
        return self.pulse_lengths[-1]
    
    """
    Adds the drive_func corresponding to a constant pulse with a sin^2
    ramp up/down to the sequence.
    t_start is offset from the end of the last pulse.
    Returns the total length of the sequence.
    """
    def const_pulse(self, wd, amp, t_pulse, t_start=0, t_rise=1):
        t_start = self.time - t_start
        def envelope(t):
                t -= t_start 
                if 0 <= t < t_rise: return amp * np.sin(np.pi*t/2/t_rise)**2
                elif t_rise <= t < t_pulse - t_rise: return amp
                elif t_pulse - t_rise <= t < t_pulse: return amp*np.sin(np.pi*(t_pulse-t)/2/t_rise)**2
                else: return 0 
        def drive_func(t, args):
            return envelope(t)*np.sin(wd*t)
        self.envelope_seq.append(envelope)
        self.pulse_seq.append(drive_func)
        self.pulse_lengths.append(t_pulse)
        self.time = t_start + t_pulse

    """
    Adds the drive_func corresponding to a gaussian pulse to the sequence.
    Returns the total length of the sequence.
    """
    def gaussian_pulse(self, wd, amp, t_pulse_sigma, t_start=0):
        t_start = self.time - t_start
        def envelope(t):
                t_max = t_start + 3*t_pulse_sigma
                return amp*gaussian(t - t_max, t_pulse_sigma)
        def drive_func(t, args):
            return envelope(t)*np.sin(wd*t)
        self.envelope_seq.append(envelope)
        self.pulse_seq.append(drive_func)
        self.pulse_lengths.append(6*t_pulse_sigma)
        self.time = t_start + 6*t_pulse_sigma
