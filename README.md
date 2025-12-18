# 🍎 Food Classification System

Hệ thống phân loại thực phẩm, hoa quả sử dụng Deep Learning với Transfer Learning và OpenCV preprocessing.

## 📋 Tổng quan

Dự án này xây dựng một classifier ảnh để phân loại trái cây, rau củ và đồ ăn sử dụng:
- **Transfer Learning**: MobileNetV2
- **Preprocessing**: OpenCV với CLAHE enhancement
- **Dataset**: Fruits-360
- **Demo**: Web application với Streamlit

## 🚀 Tính năng

- ✅ Training với MobileNet
- ✅ OpenCV preprocessing (CLAHE, denoising, augmentation)
- ✅ Validation và evaluation
- ✅ Web demo với Streamlit
- ✅ Top-K predictions
- ✅ Visualization và metrics
- ✅ Model checkpointing và early stopping

## 📦 Cài đặt

### 1. Clone repository

```bash
cd "e:\Vscode\Python\Opencv\Food Classification"
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. Chuẩn bị dữ liệu

Tổ chức dữ liệu theo cấu trúc:

```
data/
├── train/
│   ├── Apple/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── Banana/
│   │   └── ...
│   └── ...
├── validation/  (optional)
│   └── ...
└── test/
    └── ...
```

#### Download Fruits-360 Dataset

**Option 1: Kaggle API**
```bash
pip install kaggle
kaggle datasets download -d moltean/fruits
unzip fruits.zip -d data/
```

**Option 2: Manual Download**
1. Truy cập: https://www.kaggle.com/datasets/moltean/fruits
2. Download dataset
3. Extract vào thư mục `data/`

## 🎯 Sử dụng

### 1. Training

**Basic training với MobileNet:**
```bash
python train.py --model_type mobilenet --epochs 30
```

**Training với fine-tuning:**
```bash
python train.py --model_type mobilenet --epochs 30 --fine_tune --fine_tune_epochs 20
```

**Các tham số khác:**
```bash
python train.py \
    --train_dir data/train \
    --val_dir data/validation \
    --model_type mobilenet \
    --epochs 30 \
    --batch_size 32 \
    --fine_tune \
    --fine_tune_epochs 20 \
    --unfreeze_layers 30
```

### 2. Evaluation

```bash
python evaluate.py --model models/mobilenet_20241110_120000.h5 --test_dir data/test
```

### 3. Web Demo

```bash
streamlit run app_enhanced.py
```

Sau đó mở trình duyệt tại: http://localhost:8501

## 📊 Cấu trúc dự án

```
Food Classification/
├── config.py              # Cấu hình chung
├── preprocessing.py       # OpenCV preprocessing
├── data_loader.py        # Data loading và preparation
├── model.py              # Model architecture
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── app.py                # Streamlit web app
├── requirements.txt      # Dependencies
├── README.md            # Documentation
├── data/                # Dataset
│   ├── train/
│   ├── validation/
│   └── test/
├── models/              # Saved models
│   ├── *.h5
│   └── *_classes.json
└── results/             # Training results
    ├── logs/
    ├── *.png
    └── *.json
```

## 🔧 Cấu hình

Chỉnh sửa `config.py` để thay đổi các tham số:

```python
# Model configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
MODEL_TYPE = 'mobilenet'

# OpenCV preprocessing
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)

# Data augmentation
ROTATION_RANGE = 20
HORIZONTAL_FLIP = True
ZOOM_RANGE = 0.2
```

## 📈 Kết quả

Sau khi training, bạn sẽ có:

1. **Model file** (`.h5`): Saved model
2. **Class names** (`_classes.json`): Danh sách classes
3. **Training history** (`.png`): Biểu đồ accuracy/loss
4. **Training info** (`.json`): Metadata và metrics
5. **TensorBoard logs**: Trong `results/logs/`

### Xem TensorBoard

```bash
tensorboard --logdir results/logs
```

## 🎨 OpenCV Preprocessing

Các kỹ thuật preprocessing được sử dụng:

1. **Resize**: Chuẩn hóa kích thước ảnh
2. **CLAHE**: Contrast Limited Adaptive Histogram Equalization
3. **Denoising**: Non-local Means Denoising (optional)
4. **Normalization**: Scale về [0, 1]
5. **Augmentation**: Rotation, flip, brightness/contrast

## 🧪 Testing

Test preprocessing:
```bash
python preprocessing.py
```

Test data loader:
```bash
python data_loader.py
```

Test model creation:
```bash
python model.py
```

## 📱 Web Demo Features

- 📤 Upload ảnh hoặc chụp từ camera
- 🔍 Real-time classification
- 📊 Top-K predictions với confidence scores
- 📈 Visualization với bar charts
- ⚙️ Tùy chỉnh preprocessing options

## 🤝 Đóng góp

Contributions are welcome! Vui lòng:

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

## 📝 License

MIT License - xem file LICENSE để biết thêm chi tiết.

## 👨‍💻 Tác giả

Nam Lav

## 🙏 Acknowledgments

- **Fruits-360 Dataset**: https://www.kaggle.com/datasets/moltean/fruits
- **TensorFlow/Keras**: Deep learning framework
- **OpenCV**: Computer vision library
- **Streamlit**: Web app framework

## 📞 Liên hệ

Nếu có câu hỏi hoặc vấn đề, vui lòng tạo issue trên GitHub.

---

**Happy Coding! 🚀**
