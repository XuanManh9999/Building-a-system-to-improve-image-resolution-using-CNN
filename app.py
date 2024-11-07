import os
import time
from flask import Flask, request, render_template, send_file
from PIL import Image
import torch
from torchvision.transforms import ToTensor, ToPILImage
from model.srcnn import load_model, enhance_image
import math
app = Flask(__name__)

# Đường dẫn mô hình đã huấn luyện sẵn
MODEL_PATH = "model/srcnn_x3.pth"
model = load_model(MODEL_PATH)

# Đảm bảo các thư mục lưu trữ ảnh đã tồn tại
UPLOAD_FOLDER = 'static/uploads/'
ENHANCED_FOLDER = 'static/enhanced/'

# Kiểm tra và tạo thư mục nếu chưa tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ENHANCED_FOLDER, exist_ok=True)

# Giới hạn kích thước tệp ảnh tối đa (10 MB)
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Chỉ chấp nhận định dạng ảnh .png, .jpg, .jpeg
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_image(img, max_size=1024):
    """Giảm kích thước ảnh sao cho chiều dài hoặc chiều rộng không vượt quá max_size"""
    ratio = max_size / float(max(img.size))  # Tính tỷ lệ thu nhỏ
    new_size = tuple([int(x * ratio) for x in img.size])  # Tính kích thước mới
    img = img.resize(new_size, Image.BICUBIC)
    print(f"Ảnh đã được thay đổi kích thước thành: {img.size}")
    return img

# Chỉnh sửa phần tính toán PSNR giữa ảnh gốc và ảnh lý tưởng
def psnr(img1, img2):
    # Tính MSE (Mean Squared Error)
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100  # Perfect match
    return 20 * torch.log10(255.0 / torch.sqrt(mse))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")

        if not file:
            return render_template("index.html", error="Không có tệp được chọn.")
        
        if file and allowed_file(file.filename):
            if len(file.read()) > MAX_CONTENT_LENGTH:
                return render_template("index.html", error="Tệp quá lớn. Vui lòng tải lên ảnh nhỏ hơn 10MB.")
            
            file.seek(0)
            original_img_filename = file.filename
            original_img_path = os.path.join(UPLOAD_FOLDER, original_img_filename)
            file.save(original_img_path)

            start_time = time.time()

            # Mở ảnh và chuyển đổi thành grayscale
            img = Image.open(original_img_path).convert('L')
            img = resize_image(img, max_size=1024)
            img_tensor = ToTensor()(img).unsqueeze(0).to(device)

            # Tạo ảnh lý tưởng (ảnh có giá trị tối đa pixel là 255) - Không cần thiết nữa
            # ideal_img_tensor = torch.ones_like(img_tensor) * 255.0  # Không dùng

            # Chuyển ảnh qua mô hình SRCNN để cải thiện
            with torch.no_grad():
                enhanced_img = model(img_tensor)

            # Chuyển ảnh tensor thành ảnh PIL
            enhanced_img = enhanced_img.squeeze(0)
            enhanced_img = ToPILImage()(enhanced_img)

            # Lưu ảnh đã cải thiện vào thư mục
            enhanced_img_filename = f"enhanced_{original_img_filename}"
            enhanced_img_path = os.path.join(ENHANCED_FOLDER, enhanced_img_filename)
            enhanced_img.save(enhanced_img_path)

            end_time = time.time()

            # Thông tin ảnh gốc và ảnh đã cải thiện
            original_img_size = img.size
            enhanced_img_size = enhanced_img.size
            enhancement_time = round(end_time - start_time, 2)
            resolution_ratio = round((enhanced_img_size[0] / original_img_size[0]) * (enhanced_img_size[1] / original_img_size[1]), 2)

            # Tính toán PSNR giữa ảnh gốc và ảnh đã cải thiện
            original_tensor = ToTensor()(img).unsqueeze(0).to(device)
            enhanced_tensor = ToTensor()(enhanced_img).unsqueeze(0).to(device)
            psnr_value = psnr(original_tensor, enhanced_tensor)  # PSNR giữa ảnh gốc và ảnh đã cải thiện

            # Truyền thông tin vào template
            return render_template("index.html", 
                                   original_img=original_img_path, 
                                   enhanced_img=enhanced_img_path,
                                   original_img_size=original_img_size,
                                   enhanced_img_size=enhanced_img_size,
                                   enhancement_time=enhancement_time,
                                   resolution_ratio=resolution_ratio,
                                   psnr_value=round(psnr_value.item(), 2),  # Lấy giá trị PSNR
                                   show_original_info=True)
        else:
            return render_template("index.html", error="Vui lòng tải lên tệp ảnh hợp lệ (PNG, JPG, JPEG).")
    
    return render_template("index.html", original_img=None, enhanced_img=None)



if __name__ == "__main__":
    # Kiểm tra xem có GPU không, nếu có sử dụng GPU, nếu không sẽ sử dụng CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Chuyển mô hình lên GPU (nếu có)
    app.run(debug=True)
