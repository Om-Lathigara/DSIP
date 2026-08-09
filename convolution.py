import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

audio, sr = sf.read("Original.wav")

if len(audio.shape) > 1:
    audio = np.mean(audio, axis=1)

audio = audio.astype(np.float32)
audio = audio / np.max(np.abs(audio))

impulse_responses = {
    "IR1": np.array([1, 0, 1], dtype=np.float32),
    "IR2": np.array([1, 0.5, 0.25], dtype=np.float32),
    "IR3": np.array([1, 0, 1, 0, 1], dtype=np.float32)
}

results = {}

print("=" * 50)
print("AUDIO CONVOLUTION AND INVERSE FILTERING")
print("=" * 50)
print("Sample Rate :", sr)
print("Duration :", len(audio) / sr, "seconds")

for name, impulse in impulse_responses.items():

    convoluted = convolve(audio, impulse, mode="same")

    convoluted = convoluted / np.max(np.abs(convoluted))

    n = len(convoluted)

    signal_fft = np.fft.rfft(convoluted)
    impulse_fft = np.fft.rfft(
        np.pad(impulse, (0, n - len(impulse)))
    )

    epsilon = 0.001

    recovered_fft = (
        signal_fft * np.conj(impulse_fft)
    ) / (
        np.abs(impulse_fft) ** 2 + epsilon
    )

    recovered = np.fft.irfft(
        recovered_fft,
        n=n
    )

    recovered = recovered / np.max(np.abs(recovered))

    results[name] = recovered

    sf.write(
        name + "_output.wav",
        recovered,
        sr
    )

    print("\n", name)
    print("Impulse Response :", impulse)
    print("Output saved as :", name + "_output.wav")

plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(audio)
plt.title("Original Audio Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.subplot(4, 1, 2)
plt.plot(results["IR1"])
plt.title("Output after IR1 [1, 0, 1]")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.subplot(4, 1, 3)
plt.plot(results["IR2"])
plt.title("Output after IR2 [1, 0.5, 0.25]")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.subplot(4, 1, 4)
plt.plot(results["IR3"])
plt.title("Output after IR3 [1, 0, 1, 0, 1]")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.savefig("convolution_comparison.png", dpi=300)
plt.show()

print("\n" + "=" * 50)
print("PROCESS COMPLETED")
print("=" * 50)
print("All three impulse responses were processed.")
print("Comparison graph saved as convolution_comparison.png")