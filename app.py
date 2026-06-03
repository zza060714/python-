import gradio as gr
import torch
from torchvision import transforms
from PIL import Image, ImageOps
from model import Net
import os

# 初始化设备和模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

model_path = "mnist_cnn.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
else:
    print(f"Warning: '{model_path}' not found. Please train the model first by running train.py")

def recognize_digit(image):
    if image is None:
        return "请提供图片或在画板上写下数字"
    
    if not os.path.exists(model_path):
        return "模型未加载，请先运行 train.py 训练模型"
        
    # 兼容 Gradio 4.x 画板返回字典的情况
    if isinstance(image, dict):
        image = image.get("composite", image.get("image", None))
        
    if image is None:
        return "未能获取到图像数据"

    # 处理透明背景（画板默认可能是透明背景RGBA）
    if image.mode == 'RGBA':
        bg = Image.new('RGB', image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg
        
    # 将输入转为灰度图
    image = image.convert("L")
    
    # 假设输入为白底黑字，进行反色处理适配MNIST的黑底白字
    image = ImageOps.invert(image)

    # 预处理
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    tensor = transform(image).unsqueeze(0).to(device)

    # 模型推理
    with torch.no_grad():
        output = model(tensor)
        prediction = output.argmax(dim=1, keepdim=True).item()
        confidence = torch.exp(output).max().item()

    return f"识别结果: {prediction}\n置信度: {confidence:.2%}"

# 创建带有画板和上传功能的界面 (使用 Blocks API 实现多Tab组合)
with gr.Blocks(title="手写数字 OCR 识别系统", theme="default") as interface:
    gr.Markdown("# 手写数字 OCR 识别系统\n基于 PyTorch 和 CNN 模型开发。你可以直接在**手写画板**上写字，或者在**上传图片**栏中上传一张包含单个 0-9 数字的图片进行测试。")
    
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("手写画板"):
                    # 使用 Sketchpad 组件提供绘图功能
                    sketchpad_input = gr.Sketchpad(type="pil", label="请在此写下数字")
                    sketch_btn = gr.Button("识别手写数字", variant="primary")
                with gr.TabItem("上传图片"):
                    image_input = gr.Image(type="pil", label="上传白底黑字图片")
                    upload_btn = gr.Button("识别上传图片", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(label="识别结果", lines=4)
            
    # 绑定按钮事件
    sketch_btn.click(fn=recognize_digit, inputs=sketchpad_input, outputs=output_text)
    upload_btn.click(fn=recognize_digit, inputs=image_input, outputs=output_text)

if __name__ == "__main__":
    interface.launch(share=False)
