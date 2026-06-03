import torch
from torchvision import transforms
from PIL import Image, ImageOps
from model import Net
import sys
import os

def predict_image(image_path, model_path="mnist_cnn.pth"):
    if not os.path.exists(model_path):
        return f"Error: Model file '{model_path}' not found. Please run train.py first."
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型并加载权重
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 数据预处理与训练时保持一致
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        # 打开图像并转为灰度图
        image = Image.open(image_path).convert('L')
        # MNIST数据集是黑底白字，如果输入的图片是白底黑字，需要反色
        # 这里假设输入是常见的白底黑字手写图片
        image = ImageOps.invert(image)
        
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            # 获取最大概率的预测结果
            prediction = output.argmax(dim=1, keepdim=True).item()
            confidence = torch.exp(output).max().item()
            
        return f"Predicted Digit: {prediction} (Confidence: {confidence:.2%})"
    except Exception as e:
        return f"Error processing image: {str(e)}"

if __name__ == '__main__':
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(predict_image(image_path))
    else:
        print("Usage: python predict.py <path_to_image>")
