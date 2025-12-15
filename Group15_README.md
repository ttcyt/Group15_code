# 聲音真偽辨識系統 (Voice Authenticity Detection System)

這個系統使用 WavLM 預訓練模型和深度學習進行AI合成語音跟真實人聲辨識（real vs fake 聲音檢測）。

## 目錄
- [系統要求](#系統要求)
- [環境安裝](#環境安裝)
- [執行程式](#執行程式)
- [重現結果](#重現結果)
- [項目結構](#項目結構)
- [使用方式](#使用方式)

---

## 系統要求

- **Python**: 3.11+
- **CUDA**: 11.8+ (可選，用於 GPU 加速)
- **FFmpeg**: 用於音頻處理
- **記憶體**: 至少 8GB RAM (建議 16GB 以上)

### 檢查系統
```bash
python --version          
nvidia-smi                
which ffmpeg              
```

---

## 環境安裝

### 1. 創建虛擬環境
```bash
python -m venv venv
source venv/bin/activate
```


### 2. 安裝所有package
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers librosa scipy soundfile pydub flask flask-cors kagglehub scikit-learn matplotlib tqdm notebook
```

### 3. 安裝 FFmpeg (macOS)
如果尚未安裝：
```bash
brew install ffmpeg
```

### 驗證安裝
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import librosa; print(f'Librosa: {librosa.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

---

## 1. 執行程式 (模型訓練)

### 使用 Jupyter Notebook

#### 啟動 Jupyter，或直接在VScode上面運行
```bash
jupyter notebook
```

#### 在瀏覽器中打開 `final-project.ipynb`

#### 依序執行 Cell：

1. **Cell 1-3**: 環境準備
   - 定義 `AudioDataset` 類別
   - 檢查設備 (CUDA/CPU)

2. **Cell 4**: 定義模型
   - `WavLMEmbedding`: 特徵提取
   - `VoiceClassifier`: 分類器


4. **Cell 5**: 訓練模型 (約 30-60 分鐘，取決於硬體)
   - 本地資料集路徑 `dataset/for-rerecorded/training`
   - 使用訓練集訓練 10 個 Epoch
   - 保存模型到 `classifier.pt`

5. **Cell 6**: 模型推理
   - 加載訓練好的模型
   - 測試單個音頻文件

6. **Cell 7**: 定義評估函數
   - 計算準確度、AUC、精準度、召回率等指標
   - 繪製 ROC 曲線

7. **Cell 8**: 評估驗證集
   - 在驗證集上評估模型性能
   - 顯示AUC與混淆矩陣

---

### 2. Demo應用
#### 額外訓練一個模型包含
  - 1. real
  - 2. fake
  - 3. noise
   三個類別的模型，加入環境音，這個模型就可應用在實際場域上，當沒有人說話時就會顯示noise

#### 啟動後端服務器
```bash
cd demo
source venv/bin/activate
python backend.py
```
輸出應該顯示：
```
 * Serving Flask app 'backend'
 * Running on http://0.0.0.0:5010
```

#### 啟動前端服務器 (新terminal)
```bash
cd demo
python -m http.server 8000
```

#### 前端
在瀏覽器中打開: `http://localhost:8000`
或是在vscode中下載live server插件，在index.html檔案中右鍵 "open with live server"

#### 使用流程
1. 點擊「Start Continuous Detection」按鈕開始錄音
2. 系統每 1.5 秒進行一次預測
3. 實時顯示預測結果 (Real/Fake/Noise)
4. 點擊「Stop Detection」停止

---

### 3. 命令行推理
 - 運行第6個cell

修改cell中的測試文件路徑：
```python
test_files = [
    'dataset/for-rerecorded/validation/real/recording15400.wav_norm_mono.wav',
    'dataset/for-rerecorded/validation/fake/recording14054.wav_norm_mono.wav'
]
for file in test_files:
    print(f"{file}: {predict(file)}")
```

---

## 重現結果

### 準備數據集

數據集已存放在 `dataset/for-rerecorded/` 目錄，結構如下：
```
dataset/for-rerecorded/
├── training/          # 訓練集 (real/ + fake/)
├── validation/        # 驗證集 (real/ + fake/)
└── testing/          # 測試集 (可選)
```

**如需下載完整數據集**，從 [Kaggle](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset) 下載並解壓到 `dataset/` 目錄。

### 訓練模型

執行 Notebook **Cell 6**，會自動訓練 10 Epoch：
```
Epoch 1 Loss: 0.6523
Epoch 2 Loss: 0.4231
...
Epoch 10 Loss: 0.1245
模型已保存！ (classifier.pt)
```

### 評估模型

執行 Notebook **Cell 8-9**，查看評估結果：
```
Accuracy: 95-97%
AUC: 97-99%
Precision: 94-96%
Recall: 92-94%
[ROC 曲線圖表]
```

*實際結果因硬體、Epoch數、學習率而異*

---

## 檔案結構

```
project/
├── final-project.ipynb          # 訓練、推理、評估 Notebook
├── demo/                        # 實時檢測應用 (3 分類: real/fake/noise)
│   ├── backend.py               # Flask 後端 (port 5010)
│   ├── index.html               # 前端界面
│   ├── script.js                # 前端邏輯
│   ├── embeding.py              # WavLM 特徵提取
│   ├── voice_classifier.py      # 分類器模型
│   ├── voice_slicer.py          # 音頻切片工具
│   └── classifier_3.pt          # 3 分類模型權重
├── dataset/                     # 數據集
│   └── for-rerecorded/
│       ├── training/            # 訓練集 (real/ + fake/ + noise_data/)
│       ├── validation/          # 驗證集 (real/ + fake/+ noise_data/)
│       └── testing/             # 測試集 (可選)
├── classifier.pt                # 2 分類模型權重
├── venv/                        # Python 虛擬環境
└── README.md                    # 使用說明
```

---

