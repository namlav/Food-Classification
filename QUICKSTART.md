# 🚀 Quick Start Guide

Hướng dẫn nhanh để bắt đầu với Food Classification project.

## ⚡ Setup nhanh (5 phút)

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Download dataset mẫu

**Fruits-360 (Recommended):**
```bash
# Sử dụng Kaggle API
pip install kaggle
kaggle datasets download -d moltean/fruits
unzip fruits.zip -d data/
```

Hoặc download thủ công từ: https://www.kaggle.com/datasets/moltean/fruits

### 3. Tổ chức dữ liệu

Đảm bảo cấu trúc như sau:
```
data/
├── train/
│   ├── Apple/
│   ├── Banana/
│   └── ...
└── test/
    ├── Apple/
    ├── Banana/
    └── ...
```

## 🎯 Training đầu tiên

### Option 1: Training nhanh (MobileNet - nhẹ, nhanh)

```bash
python train.py --model_type mobilenet --epochs 30 --batch_size 32
```

**Thời gian:** ~10-20 phút (tùy GPU/CPU)

### Option 2: Training chất lượng cao (ResNet50)

```bash
python train.py --model_type resnet50 --epochs 50 --batch_size 16 --fine_tune
```

**Thời gian:** ~30-60 phút

## 📊 Evaluation

```bash
python evaluate.py --model models/mobilenet_YYYYMMDD_HHMMSS.h5 --test_dir data/test
```

## 🌐 Chạy Web Demo

```bash
streamlit run app.py
```

Mở browser: http://localhost:8501

## 📝 Dataset nhỏ để test nhanh

Nếu chưa có dataset lớn, tạo dataset test nhỏ:

```
data/
├── train/
│   ├── Apple/      (10-20 ảnh)
│   ├── Banana/     (10-20 ảnh)
│   └── Orange/     (10-20 ảnh)
└── test/
    ├── Apple/      (5 ảnh)
    ├── Banana/     (5 ảnh)
    └── Orange/     (5 ảnh)
```

Chạy training với ít epochs:
```bash
python train.py --epochs 10
```

## 🔥 Tips

1. **GPU vs CPU**: 
   - GPU: Nhanh hơn 10-50x
   - CPU: Vẫn chạy được nhưng chậm hơn

2. **Batch Size**:
   - GPU 4GB: batch_size=16
   - GPU 8GB+: batch_size=32
   - CPU: batch_size=8

3. **Model Selection**:
   - MobileNet: Nhẹ, nhanh, accuracy tốt (~85-90%)
   - ResNet50: Nặng hơn, chậm hơn, accuracy cao hơn (~90-95%)

4. **Fine-tuning**:
   - Thêm `--fine_tune` để tăng accuracy thêm 2-5%
   - Tốn thêm thời gian training

## ❓ Troubleshooting

### Lỗi: "No training data found"
- Kiểm tra cấu trúc thư mục `data/train/`
- Đảm bảo có ít nhất 2 classes với ảnh bên trong

### Lỗi: Out of Memory
- Giảm `batch_size`: `--batch_size 8` hoặc `--batch_size 4`
- Sử dụng MobileNet thay vì ResNet50

### Lỗi: Model file not found (Web demo)
- Đảm bảo đã train model trước
- Kiểm tra file `.h5` trong thư mục `models/`

### Training quá chậm
- Sử dụng GPU nếu có
- Giảm số epochs: `--epochs 20`
- Sử dụng MobileNet

## 📚 Tài liệu đầy đủ

Xem `README.md` để biết thêm chi tiết về:
- Cấu hình nâng cao
- Preprocessing options
- Model architecture
- API documentation

## 🎓 Workflow chuẩn

```bash
# 1. Cài đặt
pip install -r requirements.txt

# 2. Chuẩn bị data
# (Download và organize dataset)

# 3. Training
python train.py --model_type mobilenet --epochs 30

# 4. Evaluation
python evaluate.py --model models/mobilenet_*.h5 --test_dir data/test

# 5. Demo
streamlit run app.py
```

## 🎉 Kết quả mong đợi

Sau khi hoàn thành, bạn sẽ có:
- ✅ Model đã train (`.h5` file)
- ✅ Accuracy report và confusion matrix
- ✅ Web app để test real-time
- ✅ Visualization của training process

---

**Chúc bạn thành công! 🚀**
