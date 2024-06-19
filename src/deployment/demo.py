#!/usr/bin/env python3

# This software may be modified and distributed under the terms of the BSD-3-Clause license.

import datetime
import yarp
import torch
import numpy as np
from torchvision.transforms import transforms as transforms
import bipedal_locomotion_framework.bindings as blf
import hde.bindings as hde

import sys
import os
import argparse
from pathlib import Path
sys.path.insert( 0, os.path.abspath("../") )
from src.XBG import models
from preprocessing.ergoCubDataset import StandardScaler
from preprocessing.syncData import LiveSosFilter

class Stack():
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.stack = []
    
    def push(self, value):
        if len(self.stack) < self.capacity:
            self.stack.append(value)      
    
    def pull(self):
        if len(self.stack) >= self.capacity:
            self.stack.pop(0)

    def update(self, value):
        self.pull()
        self.push(value)
        return self.stack
    
    def is_full(self):
        return len(self.stack) == self.capacity
    
    def get_tensor(self):
        return torch.stack(self.stack).unsqueeze(0)
    
def build_remote_control_board_driver(
    param_handler: blf.parameters_handler.IParametersHandler, local_prefix: str
):
    param_handler.set_parameter_string("local_prefix", local_prefix)
    return blf.robot_interface.construct_remote_control_board_remapper(param_handler)

class XBG():
    def __init__(self, project, run_id, model_state):
        checkpoint = torch.load(f'../../assets/trained/{project}/{run_id}/{model_state}.pt')
        # Hyperparameters
        self.project = project
        self.img_size = 224 #checkpoint['img_size']
        self.hidden_size = 512 #checkpoint['hidden_size']
        self.steps = 48 #checkpoint['steps']
        self.subsampling = 3 #checkpoint['subsampling']
        self.control_mode = 'upper_body' #checkpoint['control_mode']
        self.walking_threshold = 0.2
        self.depth_threshold = 4000 #checkpoint['depth_threshold']
        self.weights = checkpoint['model_state_dict']
        self.scaler = StandardScaler(checkpoint['scaler_mean'] , checkpoint['scaler_std'])
        self.use_depth  = True
        # transform to convert the image to tensor
        self.transform=transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize(self.img_size, antialias=True), 
            transforms.CenterCrop(self.img_size)])
    
        self.sensor_stack = Stack(self.steps)
        self.rgb_stack = Stack(self.steps)
        self.depth_stack = Stack(self.steps)

        # Initialize network
        if self.control_mode=='full_body':
            self.input_size = 46
            self.output_size = 46 
            self.s_columns = ["neck_pitch", "neck_roll", "neck_yaw", "camera_tilt", "torso_pitch", "torso_roll", "torso_yaw", 
                        "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow", "l_wrist_roll", "l_wrist_pitch",  "l_wrist_yaw", 
                        "l_thumb_add", "l_thumb_oc", "l_index_oc", "l_middle_oc", "l_ring_pinky_oc", 
                        "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow", "r_wrist_roll", "r_wrist_pitch",  "r_wrist_yaw", 
                        "r_thumb_add", "r_thumb_oc", "r_index_oc", "r_middle_oc", "r_ring_pinky_oc", 
                        "l_hip_pitch", "l_hip_roll", "l_hip_yaw", "l_knee", "l_ankle_pitch", "l_ankle_roll", 
                        "r_hip_pitch", "r_hip_roll", "r_hip_yaw", "r_knee", "r_ankle_pitch", "r_ankle_roll"]
        elif self.control_mode=='upper_body': 
            self.input_size = 32
            self.output_size = 24 
            self.s_columns = ["neck_pitch", "neck_roll", "neck_yaw", "camera_tilt",
                        "torso_pitch", "torso_roll", "torso_yaw",
                        "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow", "l_wrist_roll", "l_wrist_pitch", "l_wrist_yaw",
                        "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow", "r_wrist_roll", "r_wrist_pitch", "r_wrist_yaw"]
        self.sjoypad_columns = ["joypad_0", "joypad_1", "joypad_2"]
    
    def create_model(self):
        self.model = eval(f'models.{self.project}(self.input_size, self.hidden_size, self.output_size)')
        self.model.load_state_dict(self.weights)
        self.model.eval()
        return self.model
    
    def update_sequence(self, sensor, rgb, depth=None, device='cpu'):
        sensor = torch.tensor(sensor)
        sensor = self.scaler.transform(sensor).float()
        self.sensor_stack.update(sensor.to(device))
        
        rgb = self.transform(rgb)
        self.rgb_stack.update(rgb.to(device))

        if self.use_depth  == True:
            depth[depth>self.depth_threshold] = self.depth_threshold
            depth = depth/self.depth_threshold            
            depth = self.transform(depth)
            self.depth_stack.update(depth.to(device))

def main(xbg):
    # Before everything let use the YarpSink for the logger and the YarpClock as clock. These are functionalities
    # exposed by blf.
    if not blf.text_logging.LoggerBuilder.set_factory(
        blf.text_logging.YarpLoggerFactory("exteroceptive-behaviour-generation")
    ):
        raise RuntimeError("Unable to set the logger factory")
    if not blf.system.ClockBuilder.set_factory(blf.system.YarpClockFactory()):
        raise RuntimeError("Unable to set the clock factory")

    param_handler = blf.parameters_handler.YarpParametersHandler()
    if not param_handler.set_from_filename(
        "config/exteroceptive-behaviour-generation-options.ini"
    ):
        raise RuntimeError("Unable to load the parameters")

    dt = param_handler.get_parameter_datetime("dt")

    handler = blf.parameters_handler.YarpParametersHandler()

    # Take the path of the current file and load the configuration file
    file_path = "config/xbgCameraConfig.ini"
    if not handler.set_from_filename(str(file_path)):
        print("Error while loading the configuration file for the Camera Device")
        return

    # Initialize the camera
    camera_driver = blf.robot_interface.construct_RGBD_sensor_client(
        handler.get_group("CAMERA_DRIVER")
    )
    camera_bridge = blf.robot_interface.YarpCameraBridge()
    camera_bridge.initialize(handler.get_group("CAMERA_BRIDGE"))
    if not camera_bridge.set_drivers_list([camera_driver]):
        print("Error while setting the camera driver")
        return

    poly_drivers = {}
    poly_drivers["REMOTE_CONTROL_BOARD"] = build_remote_control_board_driver(
        param_handler=param_handler.get_group("ROBOT_CONTROL"),
        local_prefix="balancing_controller",
    )
    if not poly_drivers["REMOTE_CONTROL_BOARD"].is_valid():
        raise RuntimeError("Unable to create the remote control board driver")

    # just to wait that everything is in place
    blf.log().info("Sleep for two seconds. Just to be sure the interfaces are on.")
    blf.clock().sleep_for(datetime.timedelta(seconds=2))

    # Create the sensor bridge
    sensor_bridge = blf.robot_interface.YarpSensorBridge()
    if not sensor_bridge.initialize(param_handler.get_group("SENSOR_BRIDGE")):
        raise RuntimeError("Unable to initialize the sensor bridge")
    if not sensor_bridge.set_drivers_list(list(poly_drivers.values())):
        raise RuntimeError("Unable to set the drivers for the sensor bridge")

    # Test the sensor bridge joint reading
    if not sensor_bridge.advance():
        raise RuntimeError("Unable to advance the sensor bridge")

    are_joints_ok, joint_positions, _ = sensor_bridge.get_joint_positions()
    if not are_joints_ok:
        raise RuntimeError("Unable to get the joint positions")
    print(joint_positions)
    are_motor_ok, motor_currents, _ = sensor_bridge.get_motor_currents()
    if not are_motor_ok:
        raise RuntimeError("Unable to get the motor currents")
    print(motor_currents)
    # Coeficients from already designed filtered
    sos = np.array([[0.08636403, 0.08636403, 0.0, 1.0, -0.82727195, 0.0]])
    currents_filter_0 = LiveSosFilter(sos)    
    currents_filter_1 = LiveSosFilter(sos)    
    currents_filter_2 = LiveSosFilter(sos)    
    currents_filter_3 = LiveSosFilter(sos)    
    currents_filter_4 = LiveSosFilter(sos)    
    currents_filter_5 = LiveSosFilter(sos)    
    currents_filter_6 = LiveSosFilter(sos)    
    currents_filter_7 = LiveSosFilter(sos)    

    # set the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = xbg.create_model().to(device)
    human_port = hde.msg.BufferedPortHumanState()
    human_port.open("/xbg/joints")
    network.connect("/xbg/joints", "/walking-coordinator/humanState:i")

    walking_port = yarp.BufferedPortBottle()
    walking_port.open("/xbg/joypad")
    network.connect("/xbg/joypad", "/walking-coordinator/goal:i")


    blf.log().info("Starting the Behaviour generation and controller. Waiting for your input")
    blf.log().info("Press enter to start the Behaviour generation and controller")
    blf.log().info(f"Model Running in {device}")
    blf.log().info("Starting the XBG System")
    while True:

        tic = blf.clock().now()
        bottle = walking_port.prepare()
        bottle.clear()
        
        # Get Joint State feedback
        if not sensor_bridge.advance():
            raise RuntimeError("Unable to advance the sensor bridge")
        are_motor_ok, motor_currents, _ = sensor_bridge.get_motor_currents()
        if not are_motor_ok:
            raise RuntimeError("Unable to get the motor currents")
        motor_currents = motor_currents[[7,8,9,10,14,15,16,17]]
        for i in range(len(motor_currents)):
            motor_currents[i] = eval(f'currents_filter_{i}._process(motor_currents[{i}]) ')
        are_joints_ok, joint_positions, _ = sensor_bridge.get_joint_positions()
        if not are_joints_ok:
            raise RuntimeError("Unable to get the joint positions")
        
        # Get camera feedback
        ret, img_rgb = camera_bridge.get_color_image("realsense")
        if not ret:
            print('RGB Camera Frame not available')
            continue
        ret, img_depth = camera_bridge.get_depth_image("realsense")
        if not ret:
            print('DEPTH Camera Frame not available')
            continue

        # Model Inference
        if xbg.rgb_stack.is_full():
            reg = model(xbg.sensor_stack.get_tensor()[:, torch.arange(0,xbg.steps,xbg.subsampling), :],
                            xbg.rgb_stack.get_tensor()[:, torch.arange(0,xbg.steps,xbg.subsampling), :],
                            xbg.depth_stack.get_tensor()[:, torch.arange(0,xbg.steps,xbg.subsampling), :])
           
            reg = torch.concat([torch.zeros(1,8), reg.cpu()],dim=1)
            reg = xbg.scaler.inverse_transform(reg.cpu())

            retargeted = reg[0,8:-3].double().detach().numpy()
            human_state = human_port.prepare()
            human_state.positions = retargeted

            human_state.jointNames = xbg.s_columns
            human_port.write()

            walking_signal = reg[0,-3:].double().detach().numpy()
            walking_signal[0] = 0 if abs(walking_signal[0]) < xbg.walking_threshold else walking_signal[0]
            walking_signal[1] = 0 if abs(walking_signal[1]) < xbg.walking_threshold else walking_signal[1]
            walking_signal[2] = 0 if abs(walking_signal[2]) < xbg.walking_threshold else walking_signal[2]
            
            bottle.addFloat64(walking_signal[0])
            bottle.addFloat64(walking_signal[1])
            bottle.addFloat64(walking_signal[2])
            walking_port.write()
            joint_positions = np.append(motor_currents, joint_positions)
            joint_positions = np.append(joint_positions, walking_signal)
        else:
            joint_positions = np.append(motor_currents, joint_positions)
            joint_positions = np.append(joint_positions, [0.0,0.0,0.0])
            print(f'joint_positions: {type(joint_positions)}, {joint_positions.dtype}')

        # Update Stacks
        xbg.update_sequence(joint_positions, img_rgb, img_depth, device)

        toc = blf.clock().now()
        delta_time = toc - tic
        if delta_time:
            fps = 1/(delta_time).total_seconds()            
            print(f'FPS: {fps}\n')
            if delta_time < dt:
                blf.clock().sleep_for(dt - delta_time)
        else:            
            print(f'FPS: undefined\n')
            blf.clock().sleep_for(dt)        


if __name__ == "__main__":
    network = yarp.Network()
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project', nargs='?', const=1, default='XBGdv2',
                        help='Name of the model or Architecture')
    parser.add_argument('-m', '--model', nargs='?', const=1, default='',
                        help='Model/running ID')
    parser.add_argument('-s', '--state', nargs='?', const=1, default='best',
                        help='State of the training like epochx')
    args = vars(parser.parse_args())

    xbg = XBG(args['project'], args['model'], args['state'])
    main(xbg)