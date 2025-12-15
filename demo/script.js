let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let recordingInterval;

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusText = document.getElementById("status");
const predictionLog = document.getElementById("predictionLog");

startBtn.onclick = async () => {
    console.log("Start button clicked!");
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
        audioChunks = [];
        isRecording = true;

        // UI 更新
        startBtn.style.display = "none";
        stopBtn.style.display = "inline-block";
        statusText.textContent = "🔴 Recording...";
        statusText.style.color = "red";

        mediaRecorder.ondataavailable = e => {
            audioChunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
            if (!isRecording) return; // 如果已停止，不處理

            const blob = new Blob(audioChunks, { type: "audio/webm" });
            console.log("Recorded blob:", blob);

            const formData = new FormData();
            formData.append("file", blob, "audio.webm");

            try {
                const res = await fetch("http://localhost:5010/predict", {
                    method: "POST",
                    body: formData
                });

                if (!res.ok) throw new Error("Server error");
                const data = await res.json();
                
                // 添加預測到日誌
                const timestamp = new Date().toLocaleTimeString();
                const predItem = document.createElement("div");
                predItem.className = "prediction-item";
                predItem.textContent = `[${timestamp}] ${data.prediction.toUpperCase()}`;
                predItem.style.borderLeftColor = data.prediction === "real" ? "#4CAF50" : "#f44336";
                predictionLog.insertBefore(predItem, predictionLog.firstChild);

                // 只保留最近 20 條
                while (predictionLog.children.length > 20) {
                    predictionLog.removeChild(predictionLog.lastChild);
                }
            } catch (err) {
                const predItem = document.createElement("div");
                predItem.className = "prediction-item";
                predItem.textContent = `Error: ${err.message}`;
                predItem.style.borderLeftColor = "#ff9800";
                predictionLog.insertBefore(predItem, predictionLog.firstChild);
            }
        };

        mediaRecorder.start();
        console.log("Recording started");

        // 每 1.5 秒錄一次音頻並發送
        recordingInterval = setInterval(() => {
            if (isRecording) {
                mediaRecorder.stop();
                
                // 重新開始錄音
                audioChunks = [];
                mediaRecorder.start();
            }
        }, 1500);

    } catch (err) {
        statusText.textContent = "❌ Microphone Error: " + err.message;
        statusText.style.color = "red";
    }
};

stopBtn.onclick = () => {
    console.log("Stop button clicked!");
    
    isRecording = false;
    
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }
    
    if (recordingInterval) {
        clearInterval(recordingInterval);
    }
    
    // UI 更新
    startBtn.style.display = "inline-block";
    stopBtn.style.display = "none";
    statusText.textContent = "⏸ Stopped";
    statusText.style.color = "gray";
    
    console.log("Recording stopped");
};
