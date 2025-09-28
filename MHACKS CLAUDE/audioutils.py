import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import threading
import time
import os
from datetime import datetime


class AudioRecorder:
    def __init__(self, filename="output.wav", fs=44100, channels=1):
        """
        Initialize the audio recorder

        Args:
            filename (str): Output filename for the recording
            fs (int): Sample rate in Hz
            channels (int): Number of audio channels (1 for mono, 2 for stereo)
        """
        self.filename = filename
        self.fs = fs
        self.channels = channels
        self.recording_list = []
        self.is_recording = False
        self.recording_thread = None
        self.stream = None

    def callback(self, indata, frames, time, status):
        """Callback function for audio stream"""
        if status:
            print(f"Audio status: {status}")
        if self.is_recording:
            self.recording_list.append(indata.copy())

    def start_recording(self):
        """Start audio recording in a separate thread"""
        if self.is_recording:
            print("Recording already in progress...")
            return False

        try:
            self.recording_list = []
            self.is_recording = True

            # Start the audio stream
            self.stream = sd.InputStream(
                samplerate=self.fs,
                channels=self.channels,
                callback=self.callback
            )
            self.stream.start()

            print("Audio recording started...")
            return True

        except Exception as e:
            print(f"Error starting audio recording: {e}")
            self.is_recording = False
            return False

    def stop_recording(self):
        """Stop audio recording and save the file"""
        if not self.is_recording:
            print("No recording in progress...")
            return None

        try:
            self.is_recording = False

            # Stop and close the stream
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            if len(self.recording_list) > 0:
                # Concatenate all recorded chunks
                recording = np.concatenate(self.recording_list, axis=0)

                # Save the recording
                write(self.filename, self.fs, recording)
                print(f"Audio saved as {self.filename}")

                return {
                    'filename': self.filename,
                    'duration': len(recording) / self.fs,
                    'sample_rate': self.fs,
                    'channels': self.channels,
                    'samples': len(recording)
                }
            else:
                print("No audio data recorded")
                return None

        except Exception as e:
            print(f"Error stopping audio recording: {e}")
            return None

    def get_recording_duration(self):
        """Get current recording duration in seconds"""
        if not self.is_recording or len(self.recording_list) == 0:
            return 0.0

        total_samples = sum(len(chunk) for chunk in self.recording_list)
        return total_samples / self.fs

    def is_recording_active(self):
        """Check if recording is currently active"""
        return self.is_recording


# Convenience functions for backward compatibility
def create_recorder(filename="output.wav"):
    """Create a new AudioRecorder instance"""
    # Add timestamp to filename to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(filename)
    timestamped_filename = f"{base}_{timestamp}{ext}"

    return AudioRecorder(filename=timestamped_filename)


def start_recording(recorder):
    """Start recording with the given recorder"""
    return recorder.start_recording()


def stop_recording(recorder):
    """Stop recording and return audio info"""
    return recorder.stop_recording()


def get_duration(recorder):
    """Get current recording duration"""
    return recorder.get_recording_duration()


# Test function
# Audio analysis integration
def analyze_recorded_audio(filename):
    """Analyze recorded audio for pauses and other metrics"""
    try:
        from scipy.io import wavfile

        sr, y = wavfile.read(filename)

        if y.ndim > 1:  # convert to mono if stereo
            y = y.mean(axis=1)

        y = y.astype(np.float32)
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))  # normalize to [-1, 1]

        # Detect pauses
        pauses = 0
        threshold = 0.2
        min_pause_len = int(0.1 * sr)  # 0.1s tolerance

        is_silent = np.abs(y) < threshold
        count = 0

        for val in is_silent:
            if val:
                count += 1
            else:
                if count >= min_pause_len:
                    pauses += 1
                count = 0

        # Check for final pause
        if count >= min_pause_len:
            pauses += 1

        duration = len(y) / sr

        # Calculate additional metrics
        rms_energy = np.sqrt(np.mean(y ** 2))
        zero_crossing_rate = np.mean(np.diff(np.sign(y)) != 0)

        return {
            'pauses': pauses,
            'duration': duration,
            'sample_rate': sr,
            'total_samples': len(y),
            'rms_energy': float(rms_energy),
            'zero_crossing_rate': float(zero_crossing_rate),
            'filename': filename
        }
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return None


def test_audio():
    """Test the audio recording functionality"""
    print("Testing audio recording...")
    recorder = create_recorder("test_audio.wav")

    print("Starting recording for 5 seconds...")
    recorder.start_recording()

    time.sleep(5)

    print("Stopping recording...")
    result = recorder.stop_recording()

    if result:
        print(f"Test completed successfully!")
        print(f"File: {result['filename']}")
        print(f"Duration: {result['duration']:.2f} seconds")

        # Test audio analysis
        analysis = analyze_recorded_audio(result['filename'])
        if analysis:
            print(f"Audio analysis - Pauses: {analysis['pauses']}, Energy: {analysis['rms_energy']:.3f}")
    else:
        print("Test failed!")


if __name__ == "__main__":
    test_audio()