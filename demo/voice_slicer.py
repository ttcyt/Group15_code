import librosa
import soundfile as sf

def slice_audio(wav_path, slice_duration=2.0, overlap=0.5):
    wav, sr = librosa.load(wav_path, sr=16000)
    slice_length = int(slice_duration * sr)
    step_length = int(slice_length * (1 - overlap))
    
    slices = []
    for start in range(0, len(wav), step_length):
        end = start + slice_length
        if end > len(wav):
            break
        slices.append(wav[start:end])
    
    return slices

slices2 = slice_audio("noisy/新錄音 9.wav", slice_duration=2.0, overlap=0.5)
for i, s in enumerate(slices2):
    i = i+154
    sf.write(f"noise_data/noisy_{i}.wav", samplerate=16000, data=s)
    i = i-154