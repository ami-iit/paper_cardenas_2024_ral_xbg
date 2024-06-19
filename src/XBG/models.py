import torch
import torch.nn as nn
import torch.nn.functional as F
from torchview import draw_graph
from ultralytics.nn.tasks import DetectionModel, BaseModel
from ultralytics import YOLO

class YOLOv8DetectionAndFeatureExtractorModel(DetectionModel):
    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=False, backbone=False):  # model, input channels, number of classes
        super().__init__(cfg, ch, nc, verbose)
        if backbone:
            del self.model[10:23]
        else:
            del self.model[-1]
    
    def custom_forward(self, x):
        """
        This is a modified version of the original _forward_once() method in BaseModel,
        found in ultralytics/nn/tasks.py.
        The original method returns only the detection output, while this method returns
        both the detection output and the features extracted by the last convolutional layer.
        """
        y = []
        features = None
        for m in self.model:
            print(m.f)
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if torch.is_tensor(x):
                features = x # keep the last tensor as features
            x = m(x)  # run
            if torch.is_tensor(x):
                features = x # keep the last tensor as features
            y.append(x if m.i in self.save else None)  # save output
        if torch.is_tensor(x):
            features = x # keep the last tensor as features
        return features#, x # return features and detection output

def create_yolov8_model(model_name_or_path, backbone=False, trainable=False):

    from ultralytics.nn.tasks import attempt_load_one_weight
    from ultralytics.cfg import get_cfg
    ckpt = None
    if str(model_name_or_path).endswith('.pt'):
        weights, ckpt = attempt_load_one_weight(model_name_or_path)
        cfg = ckpt['model'].yaml
    else:
        cfg = model_name_or_path
    model = YOLOv8DetectionAndFeatureExtractorModel(cfg, backbone=backbone)
    if weights:
        model.load(weights)
    args = get_cfg(overrides={'model': model_name_or_path})
    model.args = args  # attach hyperparameters to model

    for param in model.parameters():
        param.requires_grad = trainable
    return model

def create_depth_feature_extractor(version):
    if version==1:
        depth_fx = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5,5), stride=(1,1)), # 220
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5,5), stride=(1,1)), # 216
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1)), # 214
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # 107
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1)), # 105
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1)), # 103
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # 51
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1)), # 49
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # 24
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(1,1)), # 22
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1)), # 20
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # 10
            nn.ReLU()
            )
    if version==2:
        depth_fx = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=1), # 224
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=1), # 224
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # ============================= 112
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=1), # 112
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=1), # 112
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # ============================== 56
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=1), # 56
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # =============================== 28
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1), # 28
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # =============================== 14
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1), # 14
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # =============================== 7
            nn.ReLU()
            )
    if version==3:
        depth_fx = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=1), # 224
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=1), # 224
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # ============================= 112
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=1), # 112
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=1), # 112
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # ============================== 56
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=1), # 56
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # =============================== 28
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=1), # 28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # =============================== 14
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1), # 14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # =============================== 7
            )
    return depth_fx

class XBG(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Vision feature extraction
        self.vision_fx = create_yolov8_model("yolov8n.pt", trainable=False)

        # Sensors feature extraction
        self.sensor_fx = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        # Recursive Component
        self.temporal_component = nn.LSTM(256*7*7 + self.hidden_size, self.hidden_size, batch_first=True)

        # Output layer
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, 2*self.hidden_size),
            nn.ReLU(),
            nn.Linear(2*self.hidden_size, 2*self.hidden_size),
            nn.ReLU(),
            nn.Linear(2*self.hidden_size, self.output_size)
        )
    
class XBGrgb(XBG):
    def __init__(self, input_size, hidden_size, output_size, fusion_kernel_size=7, drop_rate=0.15):
        super().__init__(input_size, hidden_size, output_size)
        # vision feature extraction
        self.vision_fx = create_yolov8_model('yolov8n.pt', backbone=True)
        
        # print('Fusion Component')   
        self.fusion_component = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=fusion_kernel_size, stride=(1,1)),
            nn.ReLU()
        )
        fusion_output_size = ((7-fusion_kernel_size+1)**2 + 1)*self.hidden_size
        # Recursive Component
        self.temporal_component = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.LSTM(fusion_output_size, self.hidden_size, batch_first=True)
        )
        # Output layer
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, sensors, rgb_imgs, depth_imgs, *args):
        batch_size, time_steps, channels, height, width = rgb_imgs.size()
        rgb_imgs = rgb_imgs.view(-1, channels, height, width) # Reshape to (batch_size * time_steps, channels, height, width)

        # print('\nVision feature extraction')
        rgb = self.vision_fx(rgb_imgs)

        # print('\nSensors feature extraction')
        sensors = sensors.reshape(-1, self.input_size)
        sensors = self.sensor_fx(sensors)
        sensors = sensors.reshape(batch_size, time_steps, -1)
        
        # print('\nModal Fusion')
        x = self.fusion_component(rgb)
        x = x.reshape(batch_size, time_steps, -1) 
        x = torch.cat((x, sensors), dim=2)

        # print('\nRecursive Component')   
        x, _ = self.temporal_component(x)
        lstm_last_output = x[:, -1, :] # Get the last output of LSTM for each sequence

        # print('\nOutput layer')
        x = self.head(lstm_last_output)

        return x

class XBGfx(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Vision feature extraction
        self.fx = create_yolov8_model("yolov8n.pt")
        for param in self.fx.parameters():
            param.requires_grad = False

        # Sensors feature extraction
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)

        # Recursive Component
        self.lstm = nn.LSTM(256*7*7 + self.hidden_size, self.hidden_size, batch_first=True)

        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 2*self.hidden_size),
            nn.ReLU(),
            nn.Linear(2*self.hidden_size, 2*self.hidden_size),
            nn.ReLU(),
            nn.Linear(2*self.hidden_size, self.output_size)
        )
        # print(self.fc._get_name )
    
    def forward(self, sensors, rgb_imgs, *args):

        batch_size, time_steps, channels, height, width = rgb_imgs.size()
        rgb_imgs = rgb_imgs.view(-1, channels, height, width) # Reshape to (batch_size * time_steps, channels, height, width)

        # print('Vision feature extraction')
        img = self.fx.custom_forward(rgb_imgs)

        # print('Sensors feature extraction')
        sensors = sensors.reshape(-1, self.input_size)
        sensors = self.fc1(sensors)
        sensors = F.relu(self.fc2(sensors))
        sensors = sensors.reshape(batch_size, time_steps, -1)

        # print('Recursive Component')   
        x = torch.cat((img, sensors), dim=2)
        x, _ = self.lstm(x)

        # Get the last output of LSTM for each sequence
        lstm_last_output = x[:, -1, :]

        # print('Output layer')
        x = self.fc(lstm_last_output)

        return x
    
class XBGdv1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Vision feature extraction
        self.fx = create_yolov8_model("yolov8n.pt")
        for param in self.fx.parameters():
            param.requires_grad = False

        # Depth feature extraction
        self.fxd = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5,5), stride=(1,1)), # 220
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5,5), stride=(1,1)), # 216
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1)), # 214
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # 107
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1)), # 105
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1)), # 103
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # 51
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1)), # 49
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # 24
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(1,1)), # 22
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1)), # 20
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # 10
            nn.ReLU()
        )

        # Sensors feature extraction
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)

        # Recursive Component
        self.lstm = nn.LSTM(256*7*7 + 256*10*10 + self.hidden_size, self.hidden_size, batch_first=True)

        # Output layer
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, 2*self.hidden_size),
            nn.ReLU(),
            nn.Linear(2*self.hidden_size, 2*self.hidden_size),
            nn.ReLU(),
            nn.Linear(2*self.hidden_size, self.output_size)
        )
        # print(self.fc._get_name )

    def forward(self, sensors, rgb_imgs, depth_imgs, *args):

        batch_size, time_steps, channels, height, width = rgb_imgs.size()
        rgb_imgs = rgb_imgs.view(-1, channels, height, width) # Reshape to (batch_size * time_steps, channels, height, width)

        # print('Vision feature extraction')
        img = self.fx.custom_forward(rgb_imgs)
        img = img.reshape(batch_size, time_steps, -1) # 65536

        # print('Depth feature extraction')
        depth_imgs = depth_imgs.view(-1, 1, height, width) 
        depth = self.fxd(depth_imgs)
        depth = depth.reshape(batch_size, time_steps, -1) # 65536

        # print('Sensors feature extraction')
        sensors = sensors.reshape(-1, self.input_size)
        sensors = F.relu(self.fc1(sensors))
        sensors = F.relu(self.fc2(sensors))
        sensors = sensors.reshape(batch_size, time_steps, -1)

        # print('Recursive Component')   
        x = torch.cat((img, depth, sensors), dim=2)
        x, _ = self.lstm(x)

        # Get the last output of LSTM for each sequence
        lstm_last_output = x[:, -1, :]

        # print('Output layer')
        x = self.head(lstm_last_output)

        return x

class XBGdv2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Vision feature extraction
        self.fx = create_yolov8_model("yolov8n.pt")
        for param in self.fx.parameters():
            param.requires_grad = False

        # Depth feature extraction
        self.fxd = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=1), # 224
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=1), # 224
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # ============================= 112
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=1), # 112
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=1), # 112
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # ============================== 56
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=1), # 56
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # =============================== 28
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1), # 28
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # =============================== 14
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1), # 14
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # =============================== 7
            nn.ReLU()
        )
        
        self.flat_conv = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(7,7), stride=(1,1))

        # Sensors feature extraction
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)

        # Recursive Component
        self.lstm = nn.LSTM(1024 + self.hidden_size, self.hidden_size, batch_first=True)

        # Output layer
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, 2*self.hidden_size),
            nn.ReLU(),
            nn.Linear(2*self.hidden_size, self.output_size)
        )
        # print(self.fc._get_name )
    
    def forward(self, sensors, rgb_imgs, depth_imgs, *args):

        batch_size, time_steps, channels, height, width = rgb_imgs.size()
        rgb_imgs = rgb_imgs.view(-1, channels, height, width) # Reshape to (batch_size * time_steps, channels, height, width)

        # print('\nVision feature extraction')
        img = self.fx.custom_forward(rgb_imgs)
   
        # print('\nDepth feature extraction')
        depth_imgs = depth_imgs.view(-1, 1, height, width) 
        depth = self.fxd(depth_imgs)
        
        # print('\nModal Fusion')
        x = torch.cat((img, depth), dim=1)
        x = self.flat_conv(x)
        x = x.reshape(batch_size, time_steps, -1) # 65536

        # print('\nSensors feature extraction')
        sensors = sensors.reshape(-1, self.input_size)
        sensors = F.relu(self.fc1(sensors))
        sensors = F.relu(self.fc2(sensors))
        sensors = sensors.reshape(batch_size, time_steps, -1)

        # print('\nRecursive Component')   
        x = torch.cat((x, sensors), dim=2)
        x, _ = self.lstm(x)

        # Get the last output of LSTM for each sequence
        lstm_last_output = x[:, -1, :]

        # print('\nOutput layer')
        x = self.head(lstm_last_output)

        return x

class XBGdv1_1(XBG):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(input_size, hidden_size, output_size)

        # Depth feature extraction
        self.depth_fx = create_depth_feature_extractor(version=1)
        
        # print('Fusion Component')   No additional layers used for this (Just a concatenation)

        # Recursive Component
        self.temporal_component = nn.LSTM(256*7*7 + 256*10*10 + self.hidden_size, self.hidden_size, batch_first=True)
    
    def forward(self, sensors, rgb_imgs, depth_imgs, *args):
        batch_size, time_steps, channels, height, width = rgb_imgs.size()
        rgb_imgs = rgb_imgs.view(-1, channels, height, width) # Reshape to (batch_size * time_steps, channels, height, width)

        # print('\nVision feature extraction')
        rgb = self.vision_fx.custom_forward(rgb_imgs)
        rgb = rgb.reshape(batch_size, time_steps, -1) 
   
        # print('\nDepth feature extraction')
        depth_imgs = depth_imgs.view(-1, 1, height, width) 
        depth = self.depth_fx(depth_imgs)
        depth = depth.reshape(batch_size, time_steps, -1) 

        # print('\nSensors feature extraction')
        sensors = sensors.reshape(-1, self.input_size)
        sensors = self.sensor_fx(sensors)
        sensors = sensors.reshape(batch_size, time_steps, -1)
        
        # print('\nModal Fusion')
        x = torch.cat((rgb, depth, sensors), dim=2)

        # print('\nRecursive Component')   
        x, _ = self.temporal_component(x)
        lstm_last_output = x[:, -1, :] # Get the last output of LSTM for each sequence

        # print('\nOutput layer')
        x = self.head(lstm_last_output)

        return x

class XBGdv2_1(XBG):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(input_size, hidden_size, output_size)

        # Depth feature extraction
        self.depth_fx = create_depth_feature_extractor(version=2)
        
        # print('Fusion Component')   
        self.flat_conv = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(7,7), stride=(1,1))

        # Recursive Component
        self.temporal_component = nn.LSTM(1024 + self.hidden_size, self.hidden_size, batch_first=True)

        # Output layer
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, 2*self.hidden_size),
            nn.ReLU(),
            nn.Linear(2*self.hidden_size, self.output_size)
        )
    
    def forward(self, sensors, rgb_imgs, depth_imgs, *args):
        batch_size, time_steps, channels, height, width = rgb_imgs.size()
        rgb_imgs = rgb_imgs.view(-1, channels, height, width) # Reshape to (batch_size * time_steps, channels, height, width)

        # print('\nVision feature extraction')
        rgb = self.vision_fx.custom_forward(rgb_imgs)
   
        # print('\nDepth feature extraction')
        depth_imgs = depth_imgs.view(-1, 1, height, width) 
        depth = self.depth_fx(depth_imgs)

        # print('\nSensors feature extraction')
        sensors = sensors.reshape(-1, self.input_size)
        sensors = self.sensor_fx(sensors)
        sensors = sensors.reshape(batch_size, time_steps, -1)
        
        # print('\nModal Fusion')
        x = torch.cat((rgb, depth), dim=1)
        x = self.flat_conv(x)
        x = x.reshape(batch_size, time_steps, -1) 
        x = torch.cat((x, sensors), dim=2)

        # print('\nRecursive Component')   
        x, _ = self.temporal_component(x)
        lstm_last_output = x[:, -1, :] # Get the last output of LSTM for each sequence

        # print('\nOutput layer')
        x = self.head(lstm_last_output)

        return x

class XBGdv2_2(XBG):
    def __init__(self, input_size, hidden_size, output_size, fusion_kernel_size=7, drop_rate=0.15):
        super().__init__(input_size, hidden_size, output_size)

        # vision feature extraction
        self.vision_fx = create_yolov8_model('yolov8n.pt', backbone=True)
       
        # Depth feature extraction
        self.depth_fx = create_depth_feature_extractor(version=3)
        
        # print('Fusion Component')   
        self.fusion_component = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=fusion_kernel_size, stride=(1,1)),
            nn.ReLU()
        )
        fusion_output_size = ((7-fusion_kernel_size+1)**2 + 1)*self.hidden_size
        # Recursive Component
        self.temporal_component = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.LSTM(fusion_output_size, self.hidden_size, batch_first=True)
        )
        # Output layer
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size)
        )
    
    def forward(self, sensors, rgb_imgs, depth_imgs, *args):
        batch_size, time_steps, channels, height, width = rgb_imgs.size()
        rgb_imgs = rgb_imgs.view(-1, channels, height, width) # Reshape to (batch_size * time_steps, channels, height, width)

        # print('\nVision feature extraction')
        rgb = self.vision_fx(rgb_imgs)
   
        # print('\nDepth feature extraction')
        depth_imgs = depth_imgs.view(-1, 1, height, width) 
        depth = self.depth_fx(depth_imgs)

        # print('\nSensors feature extraction')
        sensors = sensors.reshape(-1, self.input_size)
        sensors = self.sensor_fx(sensors)
        sensors = sensors.reshape(batch_size, time_steps, -1)
        
        # print('\nModal Fusion')
        x = torch.cat((rgb, depth), dim=1)
        x = self.fusion_component(x)
        x = x.reshape(batch_size, time_steps, -1) 
        x = torch.cat((x, sensors), dim=2)

        # print('\nRecursive Component')   
        x, _ = self.temporal_component(x)
        lstm_last_output = x[:, -1, :] # Get the last output of LSTM for each sequence

        # print('\nOutput layer')
        x = self.head(lstm_last_output)

        return x

class XBGdepth(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, fusion_kernel_size=7, drop_rate=0.15):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Depth feature extraction
        self.depth_fx = create_depth_feature_extractor(version=3)
        
        # Sensors feature extraction
        self.sensor_fx = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        # print('Fusion Component')   
        self.fusion_component = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=fusion_kernel_size, stride=(1,1)),
            nn.ReLU()
        )
        fusion_output_size = ((7-fusion_kernel_size+1)**2 + 1)*self.hidden_size
        # Recursive Component
        self.temporal_component = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.LSTM(fusion_output_size, self.hidden_size, batch_first=True)
        )
        # Output layer
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size)
        )
    
    def forward(self, sensors, rgb_imgs, depth_imgs, *args):
        batch_size, time_steps, channels, height, width = depth_imgs.size()

        # print('\nDepth feature extraction')
        depth_imgs = depth_imgs.view(-1, 1, height, width) 
        depth = self.depth_fx(depth_imgs)

        # print('\nSensors feature extraction')
        sensors = sensors.reshape(-1, self.input_size)
        sensors = self.sensor_fx(sensors)
        sensors = sensors.reshape(batch_size, time_steps, -1)
        
        # print('\nModal Fusion')
        x = self.fusion_component(depth)
        x = x.reshape(batch_size, time_steps, -1) 
        x = torch.cat((x, sensors), dim=2)

        # print('\nRecursive Component')   
        x, _ = self.temporal_component(x)
        lstm_last_output = x[:, -1, :] # Get the last output of LSTM for each sequence

        # print('\nOutput layer')
        x = self.head(lstm_last_output)

        return x
    
if __name__== "__main__":   
    input_size = 24
    hidden_size = 512
    output_size = 24
    batch_size = 1
    steps = 16
    fusion_kernel_size = 7
    
    rgb_imgs = torch.rand(size=(batch_size, steps, 3, 224, 224))
    mask_layer = torch.rand(size=(steps, batch_size, 224, 224))
    pose_layer = torch.rand(size=(steps, batch_size, 224, 224))
    depth_imgs = torch.rand(size=(batch_size, steps, 1, 224, 224))
    sensors = torch.rand(size=(batch_size, steps, input_size)) 

    # model = XBGrgb(input_size, hidden_size, output_size)
    # model_graph = draw_graph(model, input_data=(sensors, rgb_imgs, depth_imgs), 
    #                          expand_nested=False, save_graph=True, filename='XBGrgb')
    # model = XBGdv1(input_size, hidden_size, output_size)
    # model_graph = draw_graph(model, input_data=(sensors, rgb_imgs, depth_imgs), 
    #                          expand_nested=False, save_graph=True, filename='XBGdv1')
    # model = XBGdv2_1(input_size, hidden_size, output_size)
    # model_graph = draw_graph(model, input_data=(sensors, rgb_imgs, depth_imgs), 
    #                          expand_nested=False, save_graph=True, filename='XBGdv2_1')
    model = XBGdv2_2(input_size, hidden_size, output_size, fusion_kernel_size=fusion_kernel_size)
    # model_graph = draw_graph(model, input_data=(sensors, rgb_imgs, depth_imgs), 
                            #  expand_nested=False, save_graph=True, filename='XBGdv2_2')
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{pytorch_total_train_params}/{pytorch_total_params} (Trainable/Total) parameters")