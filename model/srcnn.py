import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        
        # 1 kênh đầu vào cho ảnh grayscale
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4)  # 1 kênh đầu vào
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2)  # 1 kênh đầu ra
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x

def load_model(model_path):
    model = SRCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def enhance_image(image_path, model):
    # Mở ảnh và chuyển thành ảnh xám (grayscale)
    img = Image.open(image_path).convert('L')  # 'L' chuyển ảnh thành grayscale (1 kênh)

    # Thực hiện phóng đại ảnh (resize) với phương pháp BICUBIC
    img = img.resize((img.width * 2, img.height * 2), Image.BICUBIC)
    print(f"Image resized to: {img.size}")

    # Chuyển ảnh xám thành tensor và thêm chiều batch
    img = ToTensor()(img).unsqueeze(0)
    print(f"Image tensor shape before model: {img.shape}")

    with torch.no_grad():
        try:
            enhanced_img = model(img)  # Cải thiện ảnh bằng mô hình
            print(f"Model output shape: {enhanced_img.shape}")
        except Exception as e:
            print(f"Error during model inference: {e}")
            return None
        
    # Loại bỏ chiều batch và chuyển lại ảnh tensor thành ảnh PIL
    enhanced_img = enhanced_img.squeeze(0)
    enhanced_img = ToPILImage()(enhanced_img)
    return enhanced_img

