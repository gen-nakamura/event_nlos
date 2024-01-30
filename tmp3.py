import imageio
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
# import cv2
import numpy as np
import glob
# import re
import rosbag

rosbag_files = sorted(glob.glob("/home/gen0401/Documents/research/codes/event_nlos/Dataset/rosbag/*.bag"))
rosbag_file = rosbag_files[0]
srcpath = glob.glob("./Dataset/src/*.txt")[0]
TOPIC_EVENTS = '/cam0/events'
SIZE_X = 640
SIZE_Y = 480
num_bins = [30, 150]
FRAMES = 150


def save_images_as_movie(filename, images, fps, ffmpeg_args='-y', ffmpeg_executable='ffmpeg'):
    """画像列を動画ファイルとして保存する.

    Parameters
    ----------
    filename: str
      保存先ファイル名.
    images: iterable of array-like
      動画として保存する画像列.
    fps: int
      動画のフレームレート.
    ffmpeg_args: str or iterable of str
      ffmpegの出力ファイル設定.
      例: PowerPointで再生可能かつ無圧縮にしたいなら '-vcodec rawvideo -pix_fmt yuv420p'
    ffmpeg_executable: str
      ffmpegの実行ファイルパス.

    Return
    ------
    completed_process: subprocess.CompletedProcess
      ffmpegの実行結果.

    Examples
    --------
    >>> import numpy as np
    >>> images = [np.full(shape=(128, 128), fill_value=v, dtype=np.uint8) for v in np.linspace(0, 255, 60)]
    >>> result = save_images_as_movie('example.mp4', images, fps=60)
    >>> result.check_returncode()
    """
    from subprocess import Popen, PIPE, CompletedProcess
    from cv2 import imencode

    if isinstance(ffmpeg_args, str):
        ffmpeg_args = ffmpeg_args.split()
    ffmpeg = Popen(
        args=[
            ffmpeg_executable,
            '-r', str(fps), '-f', 'image2pipe', '-i', '-',
            '-r', str(fps), *ffmpeg_args, filename,
        ],
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
    )
    try:
        for image in images:
            success, buffer = imencode('.bmp', image)
            if not success:
                raise ValueError('imencode failed')
            ffmpeg.stdin.write(buffer)
    finally:
        out, err = ffmpeg.communicate()
    return CompletedProcess(
        args=ffmpeg.args, 
        returncode=ffmpeg.returncode, 
        stdout=out, 
        stderr=err,
    )

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
    TOPIC_EVENTS = '/cam0/events'
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
    # result = np.where(x < 0, 0, np.where(x == 0, 0.5, 1))
    x_min = x.min(axis=None, keepdims=True)
    x_max = x.max(axis=None, keepdims=True)
    return (x - x_min) / (x_max - x_min)
    return result

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


def save_voxel(file_name):
        print(file_name)
        input = load_bag_file(file_name)
        # print('bag loaded')
        input = events_to_voxel_grid(input, 150, SIZE_X, SIZE_Y)
        # print('to voxel')
        # input = np.array(input, dtype=np.int32)
        input_norm = min_max(input)
        
        frames = input_norm
        print('normalized')
        video_name = './tmp/BrandSilence00000.mp4'
        video_arr = frames*255

        # from PIL import Image
        # for i in range(150):
        #     pil_img = Image.fromarray(video_arr[i].astype(np.uint8))
        #     # RGB

        #     pil_img.save(f'./img/image{str(i).zfill(3)}.png')


        save_images_as_movie(video_name, video_arr, 50)
        
        # with imageio.get_writer(video_name, fps=30) as video:
        #         for i in range(150):
        #             frame = frames[i]*255.0  # 画像を8ビット符号なし整数に変換
        #             video.append_data(frame)

        
        # np.savez_compressed(name_input, input_norm)
        # print('saved as ',name_input)
        return 

if __name__ == "__main__":
    # shutil.rmtree('./Dataset/target_data/')
    # os.mkdir('./Dataset/target_data/')

    # for rosbag_file in rosbag_files:
        # name_input = rosbag_file.replace('rosbag', 'input_data').replace('.bag', f'_{num_bin}.npz')
        # # if os.path.exists(name_input):
        # #     os.remove(name_input)
        # # if not os.path.exists(name_input):
        # print(f'creating file: {os.path.basename(name_input)}')
        # input = load_bag_file(rosbag_file)
        # input = events_to_voxel_grid(input, num_bin, SIZE_X, SIZE_Y)
        # input = np.array(input, dtype=np.int32)
        # input_norm = min_max(input)
        # np.savez_compressed(name_input, input_norm)
    save_voxel(rosbag_file)
