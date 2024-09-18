import torch

#获得模型方法一
#model = torch.hub.load("ultralytics/yolov5", "yolov5n")#从网络上下载yolov5n的模型与权重


import cv2
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn





input_size=(640, 640)

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 定义预处理步骤

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.Resize(input_size)  # 假设模型的输入尺寸是固定的，这里需要替换为实际的尺寸
])



"""#获得模型方法二
import sys

sys.path.append('Edge_yolo/ultralytics')

from models.common import DetectMultiBackend # type: ignore
model = DetectMultiBackend('Edge_yolo/ultralytics/yolov5n.pt', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#DetectMultiBackend 类被用来加载名为 'Edge_yolo/ultralytics/yolov5n.pt' 的模型权重文件

sys.path.remove('Edge_yolo/ultralytics')

"""


#获得模型方法三
from ultralytics import YOLO
model = YOLO('yolov8n.pt',task='detect') 



from tabnanny import verbose
import cv2
import matplotlib.pyplot as plt




while True:
    # 从摄像头捕获一帧
    ret, frame = cap.read()
    #这里的ret是一个bool值，如果捕获成功就是true，反之则是false
    #frame那就是这一帧的图像咯，形状为（长，宽，3）
    """   
    #print(frame.shape)
    print(type(frame))

    if not ret:
        break

    # 将摄像头捕获的帧转换为模型可以接受的格式
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#在OpenCV中，默认的颜色格式是BGR，这是基于原始图像传感器数据的颜色顺序。
                                                #然而，许多机器学习模型和深度学习框架，包括PyTorch，期望输入数据是RGB格式的。
                                                #这行代码的作用是将一个BGR格式的图像转换为RGB格式的图像。
    #print(image.shape)
    image = preprocess(image)
    #print(image.shape)
    image = torch.unsqueeze(image, 0)  # 添加批处理维度
    #print(image.shape)

    # 使用模型进行推理
    with torch.no_grad():
        prediction = model(image)

    print(type(prediction))
    prediction.show()
    

    # 处理预测结果
    # ...

    # 显示摄像头捕获的帧
    cv2.imshow('Camera', frame)

    # 按'q'退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和关闭窗口
cap.release()
cv2.destroyAllWindows()
    """

    if ret:
            # Perform prediction on the frame
            results = model(frame,verbose=False)

            print(frame)
            print("如果有一天，你会来看一看我吗？？？\n")

            # Render the predictions on the frame
            frame = results # get the first image

            print(frame)
            print("看大雪如何衰老的，你的眼睛如何融化？？？\n")
            
            # Convert the image from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the resulting frame using matplotlib
            plt.imshow(frame_rgb)
            plt.show()

        # Release the VideoCapture object
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

