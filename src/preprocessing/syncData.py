import numpy as np
import h5py
import os
import pandas as pd
import scipy.signal
        
#%% Functions definition
def __tensor_numerical_data(file_object, data, columns, timestamps):
    """
    This function calls itself until it found the "data" attribute inside the struct
    and return the data as list format. Similar to the original _populate_numerical_data() function.
    """
    
    for key, value in file_object.items():
        if not isinstance(value, h5py._hl.group.Group):
            continue
        if key == "#refs#":
            continue
        if key == "log":
            continue
        if "data" in value.keys():
            if "elements_names" in value.keys():
                elements_names_ref = value["elements_names"]
                elements_names = [
                    f"{value.name}/" + "".join(chr(c[0]) for c in value[ref])
                    for ref in elements_names_ref[0]
                ]
            data.append(np.array(value["data"]))
            timestamps.append(np.array(value["timestamps"]))
            columns.append(elements_names)
        else:
            __tensor_numerical_data(value, data, columns, timestamps)

    return data

def open_mat_file( file_name: str, data, columns, timestamps):
    with h5py.File(file_name, "r") as file:
        root_name = 'robot_logger_device'
        root_variable = file.get(root_name)
        data = __tensor_numerical_data(file[root_name], data, columns, timestamps)
    return data

def array_2_df(timestamp, data, measurement_name):
    """
    Convert NumPy arrays to a Pandas DataFrame with customizable column names.

    Parameters:
        timestamp (numpy.ndarray): An array containing timestamps or time-related data.
        data (numpy.ndarray): The data to be converted into a DataFrame.
        measurement_name (str): The name to be assigned to the DataFrame columns.

    Returns:
        pandas.DataFrame: A DataFrame containing the provided data with columns named
        according to the 'measurement_name' parameter, and an additional 'timestamp' column.
    """
    data = data.reshape(data.shape[0], -1)    
    df = pd.DataFrame(data, columns=measurement_name)
    df['timestamp'] = np.squeeze(timestamp, 1)
    
    return df

# https://www.samproell.io/posts/yarppg/yarppg-live-digital-filter/
class LiveSosFilter():
    """Live implementation of digital filter with second-order sections.
    """
    def __init__(self, sos):
        """Initialize live second-order sections filter.

        Args:
            sos (array-like): second-order sections obtained from scipy
                filter design (with output="sos").
        """
        self.sos = sos

        self.n_sections = sos.shape[0]
        self.state = np.zeros((self.n_sections, 2))
    def _process(self, x):
        """Filter incoming data with cascaded second-order sections.
        """
        for s in range(self.n_sections):  # apply filter sections in sequence
            b0, b1, b2, a0, a1, a2 = self.sos[s, :]

            # compute difference equations of transposed direct form II
            y = b0*x + self.state[s, 0]
            self.state[s, 0] = b1*x - a1*y + self.state[s, 1]
            self.state[s, 1] = b2*x - a2*y
            x = y  # set biquad output as input of next filter section.

        return y

def mat_2_csv_log(dir, recordings):
    sos = scipy.signal.iirfilter(2, Wn=3, fs=100, btype="low", ftype="butter", output='sos')
    for recording in recordings:
        data = []
        columns = []
        timestamps = []
        #%% Reading Data
        file = f'{dir}/{recording}/{recording}.mat'
        print('\n', file)
        data = open_mat_file(file, data, columns, timestamps)
        print('=== Log data read ===')
     
        #%% Dataframe Approach
        sensors_data = pd.DataFrame()
        walk_data = pd.DataFrame()
        camera_data = pd.DataFrame()
        for i in range(len(data)):  
            """
            Process data based on column names containing either 'walking' or 'camera' 
            and merges it into separate DataFrames ('walk_data', 'camera_data', or 'sensors_data') 
            based on the column name. 
            If the target DataFrame does not exist yet, it creates a new DataFrame.
            The try is implemented to catch the first iteration where is no data to be merged yet
            """      
            if 'walking' in columns[i][0]:
                try:
                    current_data = array_2_df(timestamps[i], data[i], columns[i])
                    walk_data = walk_data.merge(current_data, left_on='timestamp', right_on='timestamp')
                except:
                    walk_data = array_2_df(timestamps[i], data[i], columns[i])
            elif 'camera' in columns[i][0]:       
                img_idx = np.arange(1, len(timestamps[i])+1)
                img_idx = np.array([f'{recording}/{recording}_realsense_rgb/img_{idx}.png' for idx in img_idx])
                try:
                    current_data = array_2_df(timestamps[i], img_idx, columns[i])
                    camera_data = camera_data.merge(current_data, left_on='timestamp', right_on='timestamp')
                except:
                    camera_data = array_2_df(timestamps[i], img_idx, columns[i])
            else:
                try:
                    current_data = array_2_df(timestamps[i], data[i], columns[i])
                    sensors_data = pd.merge_asof(sensors_data, current_data, on='timestamp')

                except:
                    sensors_data = array_2_df(timestamps[i], data[i], columns[i])
        print('=== Data populated into a matrix ===')

        print('sensors_data.shape: ', sensors_data.shape)
        print('walk_data.shape: ', walk_data.shape)
        print('camera_data.shape: ', camera_data.shape)

        #%% Sync Data
        sync_data = sensors_data

        # Filtering the currents
        df_currents = sync_data.filter(regex='/robot_logger_device/motors_state/currents/._shoulder_.|/robot_logger_device/motors_state/currents/._elbow', axis=1)
        for column in df_currents.columns:
            currents_filter = LiveSosFilter(sos)    
            sync_data.loc[:, column] = [currents_filter._process(y_temp) for y_temp in df_currents[column].values]
        
        # Merge the data based on the closest timestamp
        if camera_data.shape[0]==0:
            sync_data = sync_data.dropna()
            print('No realsense camera data available')
        else:
            sync_data = pd.merge_asof(camera_data, sync_data, on='timestamp').dropna() # Droping na allows us to filter out the data recorded while the walking was unactive
            print('sync_data.shape: ', sync_data.shape)
        # return sync_data

        try:
            sync_data.to_csv(f'{dir}/csv_data/{recording}.csv')
        except:
            os.makedirs(f'{dir}/csv_data/')
            sync_data.to_csv(f'{dir}/csv_data/{recording}.csv')

if __name__== "__main__": 
    mat_2_csv_log('/home/ittbmp014lw003/Documents/data/Exteroceptive-behaviour-generation', 
                [
                    'robot_logger_device_2023_11_13_17_34_45',
                    'robot_logger_device_2023_11_13_18_05_09',
                    'robot_logger_device_2023_11_13_18_19_35',
                    'robot_logger_device_2023_11_13_19_00_44',
                    'robot_logger_device_2023_11_13_19_05_59',
                    'robot_logger_device_2024_02_13_17_29_36',
                    'robot_logger_device_2024_02_13_17_04_57',
                    'robot_logger_device_2024_02_13_16_57_36',
                    'robot_logger_device_2024_02_13_16_54_30',
                    'robot_logger_device_2024_02_13_16_47_39',
                    'robot_logger_device_2024_02_13_16_37_20',
                    'robot_logger_device_2024_03_07_09_59_55',
                    'robot_logger_device_2024_03_08_09_16_45',
                    'robot_logger_device_2024_03_08_09_32_01',
                    'robot_logger_device_2024_03_08_09_54_41',
                    'robot_logger_device_2024_05_10_13_17_15', 
                    'robot_logger_device_2024_05_10_13_08_01',
                    'robot_logger_device_2024_05_10_12_51_37'
                    ])