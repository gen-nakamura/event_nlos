import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
# import cv2
import numpy as np
import glob
# import re
import rosbag
import matplotlib.pyplot as plt

rosbag_files = glob.glob("./Dataset/rosbag/*.bag")
srcpath = glob.glob("./Dataset/src/*.txt")[0]
TOPIC_EVENTS = '/cam0/events'
SIZE_X = 640
SIZE_Y = 480
num_bins = [30, 50, 150]

def events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()
    # イベントデータをタイムスタンプでソートする
    events = events[events[:, 0].argsort()]
    timestamps = events[:, 0]

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0


    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(np.int64)
    ys = events[:, 2].astype(np.int64)
    # xs = events[:, 1].astype(np.float32)
    # ys = events[:, 2].astype(np.float32)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int64)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid

def load_bag_file(path_to_bag_file):
    """Bagファイルからのイベントデータの読み込み.

    Args:
        path_to_bag_file (str): Bagファイルへのパス.
        timestamp_range (list[int]): [読み込みを始めるタイムスタンプ, 読み込みを終わるタイムスタンプ]の形式のリスト.

    Raises:
        Exceptions.FileNotExistError: Bagファイルが存在しない場合に発生.

    Returns:
        np.ndarray: イベントデータ.
    """
    
    print('Loading events...')
    
    bag = rosbag.Bag(path_to_bag_file)

    event_list = list()
    num_events = 0
    
    for topic, msg, t in bag.read_messages():
        if topic == TOPIC_EVENTS:
            new_event_list = msg.events
            event_list.extend(new_event_list)
                
            num_events += len(new_event_list)
            print('\r%d events are loaded.' % (num_events), end='')
    
    print()
    
    events = list()
    
    for event in event_list:
        x = event.x
        y = event.y
        timestamp = event.ts.nsecs
        polarity = 1 if event.polarity else -1
        #event_array = np.zeros((SIZE_X, SIZE_Y))
        #event_array[x][y] = polarity
        event_array = [timestamp, x, y, polarity]
        events.append(event_array)
    
    events = np.array(events, dtype=np.float32)

    return events

def min_max(x):
    x_min = x.min(axis=None, keepdims=True)
    x_max = x.max(axis=None, keepdims=True)
    return (x - x_min) / (x_max - x_min)


def get_auto_keys():
    global f
    DataName = f.readline().strip()
    print("DataName = ", DataName)
    if not DataName:
        print("EOL")
        f.close()
        return
    CameraVar = list(map(float, f.readline().split(",")))
    CameraPosition = CameraVar[:3]
    CameraRotation = CameraVar[3:]

    AutoKey = []
    KeyNum = int(f.readline())
    for i in range(KeyNum):
        AutoKey.append(list(map(float, f.readline().split(","))))
    
    LightNum = int(f.readline())
    for i in range(LightNum):
        f.readline()
        # do something with light positions

    # reset_keys(SMPLX_sec)
    # キーの設定
    # for key in AutoKey:
        # set_keys(SMPLX_sec, *key)
    # set camera position
    # set light position
    # set the callback function
    # generate
    # res = render_sequence_to_movie_minimal(DataName, PKG_DIR+seq_name, MovieFolderPath+DataName, on_finished_callback)
    # print('this is response from renderer: ', res)
    # if finished, remove tr
    return AutoKey



if __name__ == "__main__":
    for num_bin in num_bins:
        for rosbag_file in rosbag_files:
            name_input = rosbag_file.replace('rosbag', 'train_data').replace('.bag', f'_{num_bin}.npz')
            # if os.path.exists(name_input):
            #     os.remove(name_input)
            if not os.path.exists(name_input):
                print(f'creating file: {os.path.basename(name_input)}')
                input = load_bag_file(rosbag_file)
                input = events_to_voxel_grid(input, num_bin, SIZE_X, SIZE_Y)
                input = np.array(input, dtype=np.int32)
                input_norm = min_max(input)
                np.savez_compressed(name_input, input_norm)

        # f = open(srcpath, "r")
        # while True:
        #     auto_keys = np.array(get_auto_keys())
        #     if not auto_keys:
        #         break
            