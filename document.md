# Hướng dẫn sử dụng Hệ thống Cải thiện Độ phân giải Ảnh với Mô hình SR CNN

Ứng dụng này sử dụng mô hình **SRCNN (Super-Resolution Convolutional Neural Network)** để cải thiện độ phân giải của ảnh (Chỉ dùng cho ảnh xám). Bạn có thể tải lên ảnh và mô hình sẽ tự động cải thiện chất lượng ảnh với độ phân giải cao hơn.

## Các bước sử dụng:

1. **Cài đặt môi trường**:
   - Cài đặt các thư viện yêu cầu với lệnh:
     ```bash
     pip install -r requirements.txt
     ```
2. **Chạy ứng dụng**:
   - Sử dụng mô hình đã được huấn luyễn sẵn: https://github.com/yjn870/SRCNN-pytorch
   - Đảm bảo mô hình `srcnn_x3.pth` đã có trong thư mục `model/`.
   - Chạy ứng dụng Flask với lệnh:
     ```bash
     python app.py
     ```

3. **Giao diện người dùng**:
   - Truy cập `http://127.0.0.1:5000/` trong trình duyệt.
   - Tải lên ảnh và nhận ảnh đã cải thiện.

## Cấu trúc thư mục

```plaintext
project_folder/
├── model/
│   └── srcnn_x3.pth   # Mô hình SRCNN đã huấn luyện
|   └── srcnn.py         # Mã của mô hình SRCNN           
├── app.py                  # Tệp ứng dụng chính
├── static/
│   ├── uploads/            # Ảnh gốc tải lên
│   └── enhanced/           # Ảnh đã cải thiện
└── templates/
    └── index.html          # Giao diện web
```
