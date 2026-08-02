import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

# Create Output Folder
os.makedirs("outputs", exist_ok=True)

# Load Audio
def load_audio(file, duration=60):
    signal, sr = librosa.load(file, sr=22050, mono=True, duration=duration)
    return signal, sr

# Normalize Signal
def normalize(signal):
    signal = signal - np.mean(signal)
    signal = signal / np.max(np.abs(signal))
    return signal

# Pearson Correlation
def pearson_corr(x, y):
    length = min(len(x), len(y))
    x = x[:length]
    y = y[:length]
    return np.corrcoef(x, y)[0, 1]

# Cross Correlation
def cross_corr(x, y):
    length = min(len(x), len(y))

    x = x[:length]
    y = y[:length]

    corr = correlate(x, y, mode="full")

    lag = np.argmax(corr) - (length - 1)

    maximum = np.max(corr)

    return corr, lag, maximum

# Load Audio Files
original, sr = load_audio("Original.mp3")
karaoke, _ = load_audio("Karaoke.mp3")
different, _ = load_audio("Different.mp3")

# Normalize
original = normalize(original)
karaoke = normalize(karaoke)
different = normalize(different)

# Audio Information
print("=" * 50)
print("AUDIO INFORMATION")
print("=" * 50)
print("Sample Rate :", sr)
print("Original Samples :", len(original))
print("Karaoke Samples :", len(karaoke))
print("Different Samples :", len(different))

# Pearson Correlation
corr_ok = pearson_corr(original, karaoke)
corr_od = pearson_corr(original, different)
corr_kd = pearson_corr(karaoke, different)

print("\n" + "=" * 50)
print("PEARSON CORRELATION")
print("=" * 50)

print(f"Original vs Karaoke   : {corr_ok:.4f}")
print(f"Original vs Different : {corr_od:.4f}")
print(f"Karaoke vs Different  : {corr_kd:.4f}")

# Cross Correlation
cross_ok, lag_ok, max_ok = cross_corr(original, karaoke)
cross_od, lag_od, max_od = cross_corr(original, different)
cross_kd, lag_kd, max_kd = cross_corr(karaoke, different)

print("\n" + "=" * 50)
print("CROSS CORRELATION")
print("=" * 50)

print("\nOriginal vs Karaoke")
print("Maximum Correlation :", max_ok)
print("Lag :", lag_ok)

print("\nOriginal vs Different")
print("Maximum Correlation :", max_od)
print("Lag :", lag_od)

print("\nKaraoke vs Different")
print("Maximum Correlation :", max_kd)
print("Lag :", lag_kd)

# -------------------------------------------------------
# Waveforms
# -------------------------------------------------------

plt.figure(figsize=(15,8))

plt.subplot(3,1,1)
plt.plot(original)
plt.title("Original Song")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.subplot(3,1,2)
plt.plot(karaoke)
plt.title("Karaoke Version")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.subplot(3,1,3)
plt.plot(different)
plt.title("Different Song")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.tight_layout()

plt.savefig("outputs/waveforms.png", dpi=300)

plt.show()

# -------------------------------------------------------
# Cross Correlation
# -------------------------------------------------------

plt.figure(figsize=(15,10))

plt.subplot(3,1,1)
plt.plot(cross_ok)
plt.title("Cross Correlation : Original vs Karaoke")

plt.subplot(3,1,2)
plt.plot(cross_od)
plt.title("Cross Correlation : Original vs Different")

plt.subplot(3,1,3)
plt.plot(cross_kd)
plt.title("Cross Correlation : Karaoke vs Different")

plt.tight_layout()

plt.savefig("outputs/cross_correlation.png", dpi=300)

plt.show()

# -------------------------------------------------------
# Correlation Matrix
# -------------------------------------------------------

matrix = np.array([
    [1, corr_ok, corr_od],
    [corr_ok, 1, corr_kd],
    [corr_od, corr_kd, 1]
])

labels = ["Original", "Karaoke", "Different"]

plt.figure(figsize=(6,5))

plt.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)

plt.xticks(range(3), labels)
plt.yticks(range(3), labels)

plt.colorbar(label="Correlation")

for i in range(3):
    for j in range(3):
        plt.text(j, i,
                 f"{matrix[i,j]:.2f}",
                 ha="center",
                 va="center",
                 fontsize=12)

plt.title("Correlation Matrix")

plt.savefig("outputs/heatmap.png", dpi=300)

plt.show()

print("\n" + "=" * 50)
print("PROCESS COMPLETED")
print("=" * 50)
print("All graphs have been saved in the outputs folder.")