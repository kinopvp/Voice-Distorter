import numpy as np
import sounddevice as sd
import time
from scipy import signal

class VoiceDistorter:
    def __init__(self, chunk_size=512, sample_rate=44100):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.pitch_shift = -0.08
        self.bit_crush_bits = 14
        self.ring_mod_freq = 12
        self.distortion_gain = 1.1
        self.noise_threshold = 0.008
        self.gate_release_time = 0.5
        self.gate_timer = 0
        self.last_active = False
        sd.default.samplerate = self.sample_rate
        sd.default.channels = 1
        sd.default.blocksize = self.chunk_size
        
    def noise_gate(self, audio, threshold):
        rms = np.sqrt(np.mean(audio**2))
        chunk_duration = len(audio) / self.sample_rate
        if rms > threshold:
            self.gate_timer = 0
            self.last_active = True
            return audio
        else:
            if self.last_active:
                self.gate_timer += chunk_duration
            if self.gate_timer > self.gate_release_time:
                self.last_active = False
                return audio * 0.02
            elif self.last_active:
                fade_factor = 1.0 - (self.gate_timer / self.gate_release_time)
                return audio * (0.02 + fade_factor * 0.98)
            else:
                return audio * 0.02
    
    def light_bit_crush(self, audio, bits):
        max_val = 2**(bits-1)
        crushed = np.round(audio * max_val) / max_val
        blend_factor = 0.3
        return blend_factor * crushed + (1 - blend_factor) * audio
    
    def subtle_ring_modulation(self, audio, freq):
        t = np.linspace(0, len(audio)/self.sample_rate, len(audio), endpoint=False)
        modulator = 0.8 + 0.2 * np.sin(2 * np.pi * freq * t)
        return audio * modulator
    
    def pitch_shift_simple(self, audio, shift_factor):
        if abs(shift_factor) < 0.01:
            return audio
        original_length = len(audio)
        if shift_factor < 0:
            new_length = int(original_length * (1 - shift_factor))
        else:
            new_length = int(original_length / (1 + shift_factor))
        if new_length > 0:
            indices = np.linspace(0, original_length - 1, new_length)
            resampled = np.interp(indices, np.arange(original_length), audio)
            if len(resampled) < original_length:
                padded = np.zeros(original_length)
                padded[:len(resampled)] = resampled
                return padded
            else:
                return resampled[:original_length]
        return audio
    
    def apply_distortion(self, audio):
        audio = self.noise_gate(audio, self.noise_threshold)
        rms = np.sqrt(np.mean(audio**2))
        if rms < self.noise_threshold * 0.2:
            return audio * 0.01
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9
        audio = self.pitch_shift_simple(audio, self.pitch_shift)
        audio = np.tanh(audio * self.distortion_gain) * 0.95
        audio = self.light_bit_crush(audio, self.bit_crush_bits)
        audio = self.subtle_ring_modulation(audio, self.ring_mod_freq)
        try:
            sos_high = signal.butter(1, 200, btype='high', fs=self.sample_rate, output='sos')
            sos_low = signal.butter(1, 4000, btype='low', fs=self.sample_rate, output='sos')
            audio = signal.sosfilt(sos_high, audio)
            audio = signal.sosfilt(sos_low, audio)
        except:
            pass
        audio = audio * 0.9
        return np.clip(audio, -1.0, 1.0)
    
    def audio_callback(self, indata, outdata, frames, time, status):
        try:
            audio_data = indata[:, 0]
            processed = self.apply_distortion(audio_data.copy())
            outdata[:, 0] = processed
        except Exception:
            outdata.fill(0)
    
    def start(self):
        print("Voice distorter active. Press Ctrl+C to stop.")
        try:
            with sd.Stream(callback=self.audio_callback,
                          channels=1,
                          samplerate=self.sample_rate,
                          blocksize=self.chunk_size,
                          latency='low'):
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopped.")
        except Exception as e:
            print(f"Error: {e}")

def main():
    try:
        distorter = VoiceDistorter()
        distorter.start()
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    main()
