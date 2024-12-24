import gradio as gr
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import librosa.display


def calculate_basic_metrics(y, sr):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    average_pitch = np.mean(pitches[pitches > 0])
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    energy = np.sum(y ** 2)
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    return {
        'Average Pitch': average_pitch,
        'Number of MFCCs': mfccs.shape[1],
        'Energy': energy,
        'Zero Crossing Rate': zero_crossing_rate,
        'Spectral Centroid': spectral_centroid
    }


def calculate_advanced_metrics(y, sr):
    metrics = {}

    f0, _, _ = librosa.pyin(y, fmin=50, fmax=4000)
    if f0 is not None:
        metrics['Average F0 (YIN)'] = np.nanmean(f0)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    metrics['Average Chroma'] = np.mean(chroma)

    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    metrics['Average Spectral Contrast'] = np.mean(spectral_contrast)

    return metrics


def generate_spectrogram(y, sr):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max), sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()


    with tempfile.NamedTemporaryFile(delete=False, suffix='.png', mode='w+b') as f:
        plt.savefig(f.name, format='png')
        plt.close()
        return f.name


def process_audio(file):
    if file is None:
        return {}, "placeholder.png"  

    sr, y = file

    if y.dtype != np.float32:
        y = y.astype(np.float32) / np.iinfo(y.dtype).max

    basic_metrics = calculate_basic_metrics(y, sr)
    advanced_metrics = calculate_advanced_metrics(y, sr)

    metrics = {**basic_metrics, **advanced_metrics}

    image_path = generate_spectrogram(y, sr)

    return metrics, image_path


iface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(label="Upload Audio"),
    outputs=["json", "image"],
    title="Speech-Scope",
    description="Speech and audio Metrics Analysis"
)

iface.launch(debug=True)
