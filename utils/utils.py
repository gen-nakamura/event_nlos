import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import cv2
import sys

class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience=5, verbose=False, path='checkpoint_model.pth'):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = np.Inf   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path

    def __call__(self, val_loss, model):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        score = -val_loss

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(val_loss, model)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            self.checkpoint(val_loss, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_loss, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss  #その時のlossを記録する


def plot_coordinates(pred, gt):
    """
    x, y座標の集合である(150, 2)の配列をプロットする関数

    Parameters:
    coordinates (numpy.ndarray): x, y座標の集合である配列
    """
    print(gt.shape)
    # 座標の取得
    x1 = pred[:, 0]
    y1 = pred[:, 1]
    x2 = gt[:, 0]
    y2 = gt[:, 1]

    # 散布図のプロット
    plt.scatter(x1, y1, label='prediction', color='red')
    plt.scatter(x2, y2, label='ground truth', color='gray')

    # グラフの設定
    plt.title('Scatter Plot of Coordinates')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)

    # グラフの表示
    plt.show()



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


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


def frames_to_video():
    import sys
    import cv2

    # encoder(for mp4)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # output file name, encoder, fps, size(fit to image size)
    video = cv2.VideoWriter('video.mp4',fourcc, 20.0, (1240, 1360))

    if not video.isOpened():
        print("can't be opened")
        sys.exit()

    for i in range(150):
        # hoge0000.png, hoge0001.png,..., hoge0090.png
        img = cv2.imread('./hoge%04d.png' % i)

        # can't read image, escape
        if img is None:
            print("can't read")
            break

        # add
        video.write(img)
        print(i)

    video.release()
    print('written')


def create_data_check(event_frames, gt, file_name, dir='./data_check/'):
    assert event_frames.shape == (150, 480, 640) and gt.shape == (150, 2)
    event_frames = event_frames*255
    rgb_file_name = f"BrandSilence{int(file_name.split('BrandSilence')[1])}"
    rgb_dir = f'/home/gen0401/Documents/research/assets/Movies/BrandSilence/{rgb_file_name}/'
    gt_dir = dir+'gt/'+file_name+'/'
    event_dir = dir+'event/'+file_name+'/'
    concat_dir = dir+'concat/'+file_name+'/'
    concat_img_array = []
    for i in range(150):
        # plot trajectory
        plt.plot(gt[i, 0], gt[i, 1])
        plt.savefig(gt_dir+f'frame_{str(i).zfill(3)}.png')
        gt_img = cv2.imread(gt_dir+f'frame_{str(i).zfill(3)}.png')

        event_img = event_frames[i]
        cv2.imwrite(event_dir+f'frame_{str(i).zfill(3)}.png', event_img)

        rgb_img = cv2.imread(f'{rgb_dir}{rgb_file_name}-{str(i).zfill(4)}.png')

        concat_img = hconcat_resize_min([rgb_img, event_img, gt_img])
        cv2.imwrite(concat_dir+f'frame_{str(i).zfill(3)}.png', concat_img)

        height, width, layers = concat_img.shape
        size = (width, height)
        concat_img_array.append(concat_img)

    out = cv2.VideoWriter(concat_dir+'data_check.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 50.0, size)

    for concat_frame in concat_img_array:
        out.write(concat_frame)
    out.release()
    print('written')
    return
