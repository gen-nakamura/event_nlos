import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
# import cv2
import numpy as np
import glob
# import re
import rosbag
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
rosbag_files = sorted(glob.glob("/home/gen0401/Documents/research/codes/event_nlos/Dataset/rosbag/*.bag"))
srcpath = glob.glob("./Dataset/src/*.txt")[0]
TOPIC_EVENTS = '/cam0/events'
SIZE_X = 640
SIZE_Y = 480
num_bins = [30, 150]
FRAMES = 150
save_path = './img/event_plot.png'

def plot_event_data(event_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # イベントデータからx, y, t, polを取得
    timestamps = event_data[:, 0]
    x_values = event_data[:, 1]
    y_values = event_data[:, 2]
    pol_values = event_data[:, 3]

    # polの値に応じて色を設定
    colors = ['r' if pol == 1 else 'b' for pol in pol_values]

    # 3D散布図をプロット
    ax.scatter(x_values, timestamps, y_values, c=colors, marker='o')

    # 軸ラベルを設定
    ax.set_xlabel('X')
    ax.set_ylabel('Timestamp')
    ax.set_zlabel('Y')

    plt.savefig(save_path)
    plt.show()


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


events = load_bag_file(rosbag_files[0])
plot_event_data(events)
