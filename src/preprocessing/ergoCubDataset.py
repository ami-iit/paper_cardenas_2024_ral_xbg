import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
from torchvision.transforms import v2

class StandardScaler:

    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        print('=== Fit data Scaler ===')
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        print('=== Scaling the data ===')
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)
    
    def inverse_transform(self, values):
        return values * (self.std + self.epsilon) + self.mean
    
    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.epsilon = self.epsilon.to(device)

class ergoCubDataset(Dataset):

    def __init__(self, dir, records, steps, lookahead=16, subsampling=1, velocities=True, fts=True, currents=True, depth=False, rgb=True, threshold=6000, control_mode='full_body', transform=None, augmentations=None, scaler=None):
        self.dir = dir
        self.steps = steps
        self.depth = depth
        self.rgb = rgb
        self.threshold = threshold
        self.velocities = velocities
        self.fts = fts
        self.currents = currents
        self.lookahead = lookahead
        self.transform  = transform
        self.augmentations  = augmentations
        self.control_mode  = control_mode
        self.subsampling  = subsampling
        CSV_FOLDER_NAME = 'csv_data'
        CAMERA_FPS = 30
        if control_mode == 'full_upper_body':
            self.retargeting_joint_list = ["neck_pitch", "neck_roll", "neck_yaw", "camera_tilt",
                                        "torso_pitch", "torso_roll", "torso_yaw",
                                        "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow", "l_wrist_roll", "l_wrist_pitch", "l_wrist_yaw",
                                         "l_thumb_add", "l_thumb_oc", "l_index_oc", "l_middle_oc", "l_ring_pinky_oc",
                                        "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow", "r_wrist_roll", "r_wrist_pitch", "r_wrist_yaw"
                                         "r_thumb_add", "r_thumb_oc", "r_index_oc", "r_middle_oc", "r_ring_pinky_oc"]
        elif control_mode == 'upper_body':
            self.retargeting_joint_list = ["neck_pitch", "neck_roll", "neck_yaw", "camera_tilt",
                                        "torso_pitch", "torso_roll", "torso_yaw",
                                        "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow", "l_wrist_roll", "l_wrist_pitch", "l_wrist_yaw",
                                        "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow", "r_wrist_roll", "r_wrist_pitch", "r_wrist_yaw"]
        
        print('=== Loading Data ===')
        print(f'Data working directory: {dir}/{CSV_FOLDER_NAME}/')
        """
        Read all the csv files in the dir given (Only the variables of interest are kept). 
        All this variables are appended in `df_full` and then this variable is used to fit the scaler of the the sensors data
        Note: These CSVs are generated but the syncronization code that is run before
        """
        if scaler == None:
            self.mode = 'train'
            self.scaler = StandardScaler()
            df_full = []
            for record in records:
                file, start, end = record.split('@')       
                df = pd.read_csv(f'{dir}/{CSV_FOLDER_NAME}/{file}.csv', index_col=0)
                start = df.index.min() if start=='' else int(start)*CAMERA_FPS
                end = df.index.max() if end=='' else int(end)*CAMERA_FPS
                df = df.loc[start:end, :].reset_index()
                
                if self.fts:
                    df_fts = df.filter(regex='/robot_logger_device/FTs/._arm_ft', axis=1)
                    df_fts.columns = df_fts.columns.str.replace('_sensor', '')
                else: 
                    df_fts = pd.DataFrame()
                
                if self.currents:
                    df_currents = df.filter(regex='/robot_logger_device/motors_state/currents/._shoulder_.|/robot_logger_device/motors_state/currents/._elbow', axis=1)
                    # print(df_currents.columns.str.replace('/robot_logger_device/motors_state/currents/', ''))
                else: 
                    df_currents = pd.DataFrame()
                
                if self.velocities:
                    df_velocities = df.filter(regex='/robot_logger_device/joints_state/velocities/', axis=1)
                else: 
                    df_velocities = pd.DataFrame()

                df_positions = df.filter(regex='/robot_logger_device/joints_state/positions/', axis=1)
                if self.control_mode != 'full_body':
                    df_positions.columns = df_positions.columns.str.replace('/robot_logger_device/joints_state/positions/', '')
                    df_positions = df_positions[self.retargeting_joint_list]

                df_joypad = df.filter(regex='/robot_logger_device/joypad/', axis=1)
                if df_joypad.shape[1] == 0:
                    df_joypad.loc[:, ['/robot_logger_device/joypad/element_0', '/robot_logger_device/joypad/element_1', '/robot_logger_device/joypad/element_2']] = 0
                
                df = pd.concat([df_fts, df_currents, df_velocities, df_positions, df_joypad], axis=1)
                df_full.append(df)

            df_full = pd.concat(df_full, axis=0)
            print('=== Missing data filled with 0 :')
            print(df_full.isna().sum()[(df_full.isna().sum() > 0)])
            df_full.fillna(0, inplace=True)
            self.scaler.fit(torch.tensor(df_full.values))
            self.input_names = df_full.columns.tolist()
            self.output_names = df_positions.columns.tolist() + df_joypad.columns.tolist()
        else:
            self.mode = 'eval'
            self.scaler = scaler
        print(f'=== Dataset for {self.mode} ===')
        
        self.targets = []
        for record in records:
            for i in range(self.subsampling):
                file, start, end = record.split('@')   
                print(f'\nReading {file}.csv From {start}s to {end}s')
                df = pd.read_csv(f'{dir}/{CSV_FOLDER_NAME}/{file}.csv', index_col=0)
                start = df.index.min() if start=='' else int(start)*CAMERA_FPS
                end = df.index.max() if end=='' else int(end)*CAMERA_FPS
                df = df.loc[start:end, :].reset_index()
                df = df.iloc[i::self.subsampling]

                if self.fts:
                    df_fts = df.filter(regex='/robot_logger_device/FTs/._arm_ft', axis=1)
                    df_fts.columns = df_fts.columns.str.replace('_sensor', '')
                else: 
                    df_fts = pd.DataFrame()

                if self.currents:
                    df_currents = df.filter(regex='/robot_logger_device/motors_state/currents/._shoulder_.|/robot_logger_device/motors_state/currents/._elbow', axis=1)
                else: 
                    df_currents = pd.DataFrame()
                
                if self.velocities:
                    df_velocities = df.filter(regex='/robot_logger_device/joints_state/velocities/', axis=1)
                else: 
                    df_velocities = pd.DataFrame()

                df_positions = df.filter(regex='/robot_logger_device/joints_state/positions/', axis=1)
                if self.control_mode != 'full_body':
                    df_positions.columns = df_positions.columns.str.replace('/robot_logger_device/joints_state/positions/', '')
                    df_positions = df_positions[self.retargeting_joint_list]

                df_joypad = df.filter(regex='/robot_logger_device/joypad/', axis=1)
                if df_joypad.shape[1] == 0:
                    df_joypad.loc[:, ['/robot_logger_device/joypad/element_0', '/robot_logger_device/joypad/element_1', '/robot_logger_device/joypad/element_2']] = 0
                print('fts, currents, velocities, positions, joypad: ')
                print(df_fts.shape, df_currents.shape, df_velocities.shape, df_positions.shape, df_joypad.shape)

                # Last {steps} samples to sensor sequence
                sensor_aux = pd.concat([df_fts, df_currents, df_velocities, df_positions, df_joypad], axis=1)
                sensor_aux = self.scaler.transform(torch.tensor(sensor_aux.values)).float()
                sensor_inputs = sensor_aux.detach().clone()
                for step in range(1, self.steps):
                    sensor_inputs = torch.cat((sensor_aux.roll(step, 0), sensor_inputs), dim=1)
                    # print(f'step {step}: {sensor_inputs.shape}')
                sensor_inputs = sensor_inputs[self.steps-1:-self.lookahead]
                sensor_inputs = sensor_inputs.reshape(-1, self.steps, sensor_aux.shape[1])
                print('=== Sensor Sequence Built ===')

                targets = sensor_aux[:, -(df_positions.shape[1]+df_joypad.shape[1]):].roll(-self.lookahead,0)
                targets = targets[self.steps-1:-self.lookahead]
                print('sensor_inputs.shape')
                print(sensor_inputs.shape)
                print('targets.shape')
                print(targets.shape)

                # Last {steps} images to make the video sequence
                rgb_inputs = pd.DataFrame()
                rgb_inputs['t'] = df.filter(regex='/robot_logger_device/camera/realsense/rgb/', axis=1) #df['/robot_logger_device/camera/realsense/rgb']
                for step in range(1, self.steps):
                    rgb_inputs[f't-{step}'] = df.filter(regex='/robot_logger_device/camera/realsense/rgb/', axis=1).shift(periods=step)  #df['/robot_logger_device/camera/realsense/rgb'].shift(periods=step) 
                rgb_inputs = rgb_inputs[self.steps-1:-self.lookahead]
                print('=== Video Sequence Built ===')

                try:
                    self.rgb_inputs = pd.concat([self.rgb_inputs, rgb_inputs], axis=0)
                    self.sensor_inputs = torch.cat((self.sensor_inputs, sensor_inputs), dim=0)
                    self.targets = torch.cat((self.targets, targets), dim=0)
                except:
                    self.rgb_inputs = pd.DataFrame(rgb_inputs)
                    self.sensor_inputs = sensor_inputs
                    self.targets = targets
    
        print('\n=== Dataset Loaded Successfully ===')

    def __len__(self):
        return len(self.targets)    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rgb_frames = []
        depth_frames = []
        for t in range(self.steps):
            rgb_img_name = os.path.join(self.dir, self.rgb_inputs.iloc[idx, t])
            if self.rgb:
                rgb_img = cv2.imread(rgb_img_name)
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

                if self.transform:
                    rgb_img = self.transform(rgb_img)
                if self.augmentations:
                    rgb_img = self.augmentations(rgb_img)
            else:
                rgb_img = torch.tensor(0)

            if self.depth:
                depth_img_name = rgb_img_name.replace('rgb', 'depth')
                depth_img = cv2.imread(depth_img_name, cv2.IMREAD_UNCHANGED)
                depth_img[depth_img>self.threshold] = self.threshold
                depth_img = depth_img/self.threshold
                # depth_img = depth_img/1.0
                if self.transform:  
                    depth_img = self.transform(depth_img).float()
                    
            else:
                depth_img = torch.tensor(0)

            rgb_frames.insert(0, rgb_img)
            depth_frames.insert(0, depth_img)

        rgb_video = torch.stack(rgb_frames, dim=0)
        depth_video = torch.stack(depth_frames, dim=0)
        sensors = self.sensor_inputs[idx]
        target = self.targets[idx]
        
        return [rgb_video, depth_video, sensors, target]
    
if __name__== "__main__":           
    batch_size = 1
    img_size = 224
    steps = 3
    lookahead =  10
    subsampling = 3
    velocities = False
    fts = False
    currents = True
    depth = True
    control_mode = 'upper_body'
    threshold = 4000
    path =  '/data/Exteroceptive-behaviour-generation'
    records = [
        'robot_logger_device_2023_11_13_17_34_45@300@420',
        'robot_logger_device_2023_11_13_18_05_09@890@',
        'robot_logger_device_2023_11_13_19_05_59@100@',
        'robot_logger_device_2024_02_13_17_29_36@@',
        'robot_logger_device_2024_02_13_17_04_57@@',
        'robot_logger_device_2024_02_13_16_57_36@37@',
        'robot_logger_device_2024_02_13_16_54_30@@',
        'robot_logger_device_2024_02_13_16_47_39@@',
        'robot_logger_device_2024_02_13_16_37_20@45@',
        'robot_logger_device_2024_03_07_09_59_55@50@640',
        'robot_logger_device_2024_03_08_09_16_45@310@',
        'robot_logger_device_2024_03_08_09_32_01@130@',
        'robot_logger_device_2024_03_08_09_54_41@13@620',
        ]
    transformations = v2.Compose([v2.ToTensor(),
                                v2.Resize(img_size, antialias=True), 
                                v2.CenterCrop(img_size)])
    augmentations = v2.RandomApply([
                        v2.RandomChoice([v2.RandomErasing(p=0.3, scale=(0.01, 0.02), value=1),
                                        v2.Compose([v2.RandomZoomOut(fill=1, side_range=(1.0, 1.05), p=0.3), v2.Resize(img_size, antialias=True)]),
                                        v2.RandomRotation((-4,4), fill=1),
                                        v2.ElasticTransform(alpha=50, sigma=5, fill=1),
                                        v2.RandomEqualize(p=0.5),
                                        v2.ColorJitter(brightness=.5, hue=.3),
                                        v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
                                        v2.RandomPosterize(4),
                                        v2.RandomAutocontrast(p=0.5)])
                                    ], 0.3)  

    train_dataset = ergoCubDataset(path,
                                records, 
                                steps=steps, lookahead=lookahead, depth=depth, velocities=velocities, fts=fts, currents=currents, subsampling=subsampling, control_mode=control_mode, 
                                augmentations=augmentations, transform=transformations)
    print(train_dataset.input_names)
    print(len(train_dataset.input_names.values))
    print(train_dataset.__len__())                              