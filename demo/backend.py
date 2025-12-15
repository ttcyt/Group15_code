from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import io
from embeding import WavLMEmbedding
from voice_classifier import VoiceClassifier
from pydub import AudioSegment

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_model = WavLMEmbedding().to(device)
classifier = VoiceClassifier().to(device)
classifier.load_state_dict(torch.load("classifier_3.pt", map_location=device))
classifier.eval()


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    audio_bytes = request.files["file"].read()
    
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))

        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)

        wav = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    except Exception as e:
        return jsonify({"error": f"Failed to process audio: {str(e)}"}), 400

    try:
        emb = embedding_model.extract(wav)  
        emb = emb.squeeze(0)  
        
        emb = emb.unsqueeze(0) 

        with torch.no_grad():
            logits = classifier(emb)
            pred = torch.argmax(logits, dim=1).item()
    except Exception as e:
        return jsonify({"error": f"Failed to extract features: {str(e)}"}), 400

    # 映射預測結果
    label_map = {0: "real", 1: "fake", 2: "noise"}
    label = label_map.get(pred, "unknown")
    
    return jsonify({"prediction": label})
    
@app.route("/")
def hello():
    return "Audio classifier backend running."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5010, debug=True)
