from flask import Flask, send_file, request, Response, render_template
from io import StringIO
from pywt import wavedec
import warnings
warnings.filterwarnings('ignore')
import joblib
import pandas as pd
from matplotlib import *
from scipy.fftpack import fft,ifft
import pyeeg
import pywt
from pylab import *
from numpy import *
import numpy as np
from numpy import array
#from keras.models import model_from_json
from itertools import groupby
from operator import itemgetter
from xgboost import XGBClassifier
import scipy
from scipy.signal import *
import scipy.signal as signal
import matplotlib.pyplot as plt
import struct
import pathlib
import scipy.fftpack
import io
import zipfile
from scipy import signal
import shap
from datetime import timedelta
from flask_httpauth import HTTPBasicAuth
import pyedflib
from werkzeug.utils import secure_filename
import os
import pickle
import datetime
from werkzeug.exceptions import HTTPException
import glob
import errno
import matplotlib
from matplotlib import patches
from flask import send_from_directory
from json import load, dump
matplotlib.use('Agg')

auth = HTTPBasicAuth()

plt.rcParams.update({'font.size': 16})

app = Flask(__name__)

# 自动重载模板文件
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['DEBUG'] = True
# 设置静态文件缓存过期时间
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
app.config['PROPAGATE_EXCEPTIONS'] = True
app.config["IMAGE_UPLOADS"] = "static/uploads/"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["EDF","CSV","PICKLE"]



class ChannelHeader:
    def __init__(self,seq,name):
        self.seq = seq
        self.name = name
        self.offset_in_chunk = -1
        self.sample_rate = -1
        #chunk_size means number of elements in one chunk
        self.chunk_size = -1


def filetime(edf_filename):
    with io.open(edf_filename, 'rb') as f:
        f.seek(168, io.SEEK_CUR)
        headinfo = str(f.read(16))
        day = int(headinfo[2:4])
        month = int(headinfo[5:7])
        year = 2000 + int(headinfo[8:10])
        hour = int(headinfo[10:12])
        minu = int(headinfo[13:15])
        sec = int(headinfo[16:18])
    return day, month, year, hour, minu, sec

# coding=utf-8
import io
import struct


# Input
#   edf_filename: full path name of EDF file
# Return
#   data_map: key=channel_name, value=data list
#   cHdrList: header info of each channel
def readRawDataFromEdf(edf_filename):
    fs = []
    with io.open(edf_filename, 'rb') as f:
        cHdrList = readHeader(f)
        # DEBUG
        chunk_size = 0
        for cHdr in cHdrList:
            # print ("%s, sample_rate=%d" % (cHdr.name, cHdr.sample_rate))
            chunk_size = chunk_size + cHdr.sample_rate
            fs.append(cHdr.sample_rate)
        # print ("total bytes in one chunk: %d" % ((2*chunk_size)))
        data_map, channel_name = readData(f, cHdrList)
        sampling = fs[1]

    return data_map, cHdrList, sampling, channel_name


# read header information from EDF file stream
# INPUT f: handle to EDF file, caller must gurantee the position is set to START of the file
# RETURN cHdrList: a list, each element represent a channel
def readHeader(f):
    f.seek(243, io.SEEK_CUR)
    chunk_time_len = int(f.read(8))
    # print chunk_time_len
    num_of_channels = int(f.read(4))
    # print num_of_channels

    # read num_of_channels*16 byte for names
    cHdrList = []
    for cid in range(num_of_channels):
        channel_name = str(f.read(16)).strip()
        cHdrList.append(ChannelHeader(cid, channel_name))

    # Skip data until sample rate
    f.seek(num_of_channels * 200, io.SEEK_CUR)

    # read sample rate of each channel
    offset = 0
    for cid in range(num_of_channels):
        chunk_size = int(f.read(8))
        cHdrList[cid].offset_in_chunk = offset
        cHdrList[cid].chunk_size = chunk_size
        cHdrList[cid].sample_rate = chunk_size / chunk_time_len

    # skip reserved field
    f.seek(num_of_channels * 32 + 1, io.SEEK_CUR)

    return cHdrList


# read data from EDF file, each data is a 16 bits integer with sign
# INPUT f: handle to EDF file, caller must gurantee the position is set to START + (num_of_channel + 1 )*256, i.e. the start position of data section
# RETURN data_map: key=channel_name, value=data list
def readData(f, cHdrList):
    # prepare the result data
    data_map = {}
    channel_name = []
    for cHdr in cHdrList:
        data_map[cHdr.name] = []
        channel_name.append(cHdr.name)

    # read each channel
    eof = False
    chunk_cnt = 0
    while True:
        for cHdr in cHdrList:
            # read data for only one channel in one chunk
            for dataIdx in range(cHdr.chunk_size):
                data_array = f.read(2)
                # DEBUG
                # print data_array.encode('hex')
                if (len(data_array) == 0):
                    eof = True
                    break
                # unpack using little-endian
                data = struct.unpack('<h', data_array)[0]
                data_map[cHdr.name].append(data)

            if eof:
                break

        chunk_cnt = chunk_cnt + 1
        if eof:
            # print ("Chunk_count = %d" % (chunk_cnt))
            break

    return data_map, channel_name


def writeDataIntoDisk(filename, dataList):
    with open(filename, 'w') as f:
        for data in dataList:
            f.write("%d," % data)


def ensureUtf(s):
    try:
        if type(s) == unicode:
            return s.encode('utf8', 'ignore')
    except:
        return str(s)


def allowed_image(filename):

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


def transform(text_file_contents):
    return text_file_contents.replace("=", ",")

def update_ticks(x, pos):
    return x/500

def getday(y,m,d,n):
    the_date = datetime.datetime(y,m,d)
    result_date = the_date + datetime.timedelta(days=n)
    d = result_date.strftime('%Y-%m-%d')
    return d

def window_stack(a, stepsize=1, width=2):
    n = a.shape[0]
    return np.hstack([a[i:1 + n + i - width:stepsize] for i in range(width)])

from itertools import groupby
from operator import itemgetter

def time(nums):
    start = []
    end = []
    arr = []
    for k, g in groupby(enumerate(nums), lambda ix: ix[0] - ix[1]):
        arr.append(list(map(itemgetter(1), g)))
    for j in range(0, len(arr)):
        if (len(arr[j]) >= 3):
            start.append(arr[j][0])
            end.append(arr[j][-1])
    return start, end

def findSmallest(arr):
    smallest = arr[0]  # 将第一个元素的值作为最小值赋给smallest
    smallest_index = 0  # 将第一个值的索引作为最小值的索引赋给smallest_index
    for i in range(1, len(arr)):
        if arr[i] < smallest:  # 对列表arr中的元素进行一一对比
            smallest = arr[i]
            smallest_index = i
    return smallest_index

def selectionSort(arr):
    newArr = []
    for i in range(len(arr)):
        smallest = findSmallest(arr)  # 一共要调用5次findSmallest
        newArr.append(arr.pop(smallest))  # 每一次都把findSmallest里面的最小值删除并存放在新的数组newArr中
    return newArr


def wrcoef(X, coef_type, coeffs, wavename, level):
    N = np.array(X).size
    a, ds = coeffs[0], list(reversed(coeffs[1:]))

    if coef_type == 'a':
        return pywt.upcoef('a', a, wavename, level=level)[:N]
    elif coef_type == 'd':
        return pywt.upcoef('d', ds[level - 1], wavename, level=level)[:N]
    else:
        raise ValueError("Invalid coefficient type: {}".format(coef_type))


def eeg_win2(data):
    cell_array_size = len(data) // 2500
    cell_array = [None] * cell_array_size
    k = 0

    for i in range(0, len(data), 2500):
        if i + 2499 < len(data):
            cell_array[k] = data[i : (i + 2499)]
        else:
            cell_array[k] = data[i:]

        k += 1

    eeg_window = np.array(cell_array)

    return eeg_window

def var(data):
    # Variance
    return np.var(data)

def energy(data):
    # Energy (sum of squared values)
    return np.sum(data**2)

def nonlin_energy(data):
    nleo = np.zeros_like(data)  # Initialize an array to store NLEO values

    for t in range(1, len(data)-1):
        nleo[t] = data[t]**2 - data[t-1] * data[t+1]  # compute non-linear energy for each sample of the data

    nl = np.sum(nleo)
    return nl

def shan_energy(data):
    data1 = np.abs(data)

    # Create bins and bin edges
    num_bins = int(np.round(np.log2(2500) + 1))
    bin_edges = np.linspace(np.min(data1), np.max(data1), num_bins + 1)

    # Compute histogram counts for each bin
    hist_counts, _ = np.histogram(data1, bins=bin_edges)

    # Compute probabilities of each bin
    prob = hist_counts / np.sum(hist_counts)

    # Eliminate non-zero probabilities to avoid log(0) issues
    prob = prob[prob > 0]

    # Compute Shannon energy
    SE = np.sum(prob * np.log2(prob))

    return SE

def featExtract_TD(data):
    # Assuming the functions shan_energy, var, energy, and nonlin_energy are defined elsewhere
    SE = [shan_energy(d) for d in data]
    var1 = [np.var(d) for d in data]
    energ = [energy(d) for d in data]
    NE = [nonlin_energy(d) for d in data]

    feat = np.column_stack((SE, var1, energ, NE))
    return feat


def wrcoef(X, coef_type, coeffs, wavename, level):
    N = np.array(X).size
    a, ds = coeffs[0], list(reversed(coeffs[1:]))

    if coef_type =='a':
        return pywt.upcoef('a', a, wavename, level=level)[:N]
    elif coef_type == 'd':
        return pywt.upcoef('d', ds[level-1], wavename, level=level)[:N]
    else:
        raise ValueError("Invalid coefficient type: {}".format(coef_type))


@app.route('/EP-Classification/')
def psr_index():

    return render_template('EPC_home.html')


@app.route('/EP-Classification/EDF')
def psr_indexedf():
    return render_template('EPC_edf.html')


@app.route('/EP-Classification/CSV-PICKLE')
def psr_indexcsv():
    return render_template('EPC_csv.html')


@app.route('/EP-Classification/EPC_help.html')
def psr_help():
    return render_template('EPC_help.html')



@app.route('/psrdownload/')
def psr_downloadFigure ():
    file_name_EP = "Feature_analysis.zip"
    return send_file(file_name_EP, as_attachment=True)

#app.route('/EP-Classification/download/')
#def psr_downloaddataset():
#    train_path = "EPC-AI_Dataset.zip"
#    return send_file(train_path, as_attachment=True)

@app.route('/psrexample/')
def psr_downloadexample ():
    example_path = "Mice_example.zip"
    return send_file(example_path, as_attachment=True)

@app.route('/EPC_result', methods=["GET", "POST"])

def psr_transform_view():
    image = request.files["image"]
    if allowed_image(image.filename):
        filename = secure_filename(image.filename)
        #print(filename)
        image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))
        #print("File saved")
    size = os.stat("static/uploads/" + filename).st_size
    #print(size)

    if filename.rsplit(".", 1)[1].upper() == 'EDF':

        f = pyedflib.EdfReader("static/uploads/" + filename)
        n = f.signals_in_file
        signal_labels = f.getSignalLabels()
        new_signal_label = []
        sigbufs = np.zeros((n, f.getNSamples()[0]))

        for p11 in np.arange(2):
            sigbufs[p11, :] = f.readSignal(p11)
        df_signals = pd.DataFrame(sigbufs)
        df_signals = df_signals.transpose()
        # df_signals.columns = signal_labels
        df_signals.columns = ['EEG1', 'EEG2']

    elif filename.rsplit(".", 1)[1].upper() == 'CSV':
        df_signals = pd.read_csv("static/uploads/" + filename)
        df_signals.columns = ['index','EEG1', 'EEG2']


    df_signals['pre_EEG1'] = df_signals['EEG1'] - mean(df_signals['EEG1'])
    df_signals['pre_EEG2'] = df_signals['EEG2'] - mean(df_signals['EEG2'])

    fs = 500
    # Notch filter
    f0 = 50.0  # Frequency to be removed from signal (Hz)
    # Q = 15.0  # Quality factor
    Q = 35
    # Design notch filter
    b, a = signal.iirnotch(f0, Q, fs)

    freq, h = signal.freqz(b, a)

    n = len(df_signals)  # length of the signal
    k = np.arange(n)
    T = n / fs
    frq = k / T  # two sides frequency range
    frq1 = frq[range(int(n / 2))]  # one side frequency range

    df_EEG1 = pd.DataFrame(signal.filtfilt(b, a, df_signals['pre_EEG1'], method='gust'))
    df_EEG2 = pd.DataFrame(signal.filtfilt(b, a, df_signals['pre_EEG2'], method='gust'))

    df_EEG1_notch = df_EEG1[0]
    df_EEG2_notch = df_EEG2[0]

    segment_C1_train = eeg_win2(df_EEG1_notch)
    segment_C2_train = eeg_win2(df_EEG2_notch)

    waveletFunction = 'db4';

    coeffs = wavedec(df_EEG1_notch, waveletFunction, level=7)

    EEG1_D2 = wrcoef(df_EEG1_notch, 'd', coeffs, waveletFunction, 2)
    EEG2_D2 = wrcoef(df_EEG2_notch, 'd', coeffs, waveletFunction, 2)

    feat_set_C1_train_TD = featExtract_TD(segment_C1_train)
    feat_set_C2_train_TD = featExtract_TD(segment_C2_train)

    df_time_C1 = pd.DataFrame(feat_set_C1_train_TD)
    df_time_C1.columns = ['time_C1_shan_energy', 'time_C1_var', 'time_C1_energy', 'time_C1_nonlin_energy']
    df_time_C2 = pd.DataFrame(feat_set_C2_train_TD)
    df_time_C2.columns = ['time_C2_shan_energy', 'time_C2_var', 'time_C2_energy', 'time_C2_nonlin_energy']

    df_ripple_c1 = pd.DataFrame(featExtract_TD(eeg_win2(EEG1_D2)))
    df_ripple_c1.columns = ['ripple_C1_shan_energy', 'ripple_C1_var', 'ripple_C1_energy', 'ripple_C1_nonlin_energy']

    df_ripple_c2 = pd.DataFrame(featExtract_TD(eeg_win2(EEG2_D2)))
    df_ripple_c2.columns = ['ripple_C2_shan_energy', 'ripple_C2_var', 'ripple_C2_energy', 'ripple_C2_nonlin_energy']

    final_TDfeat = pd.concat([df_time_C1, df_ripple_c1, df_time_C2, df_ripple_c2], axis=1)
    df_data = final_TDfeat[['time_C1_nonlin_energy', 'ripple_C1_energy', 'time_C2_nonlin_energy', 'ripple_C2_energy']]
    df_data.columns = ['Nl_ene_c1', 'Energy_rip_c1', 'Nl_ene_c2', 'Energy_rip_c2']

    ppp = 'feature_analysis/'
    pathlib.Path(ppp).mkdir(parents=True, exist_ok=True)
    my_zip = zipfile.ZipFile('Feature_analysis.zip', 'w')
    df_mean = pd.DataFrame(df_data.mean())
    df_std = pd.DataFrame(df_data.std())
    df_ana = pd.concat([df_mean, df_std], axis=1)
    df_ana.columns = ['Mean', 'STD']
    df_ana.to_csv(ppp+"Feature_analysis.csv")
    my_zip.write(ppp + "Feature_analysis.csv")


    plt.rcParams.update({'font.size': 16})

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    df_y = df_signals["EEG1"]

    ax1.plot(df_y)
    ax1.set_ylabel('µV')
    ax1.set_title('Amplitude')
    ax1.set_xticklabels(['', '', '', '', '', ''])

    vmin = 20 * np.log10(np.max(df_y)) - 58.1  # hide anything below -40 dBc
    ax2.specgram(df_y, Fs=500, cmap=cm.jet, vmin=vmin);
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('Freq (Hz)');
    ax2.set_xlabel('Time (sec)');
    ax2.set_title('Spectrogram');


    fig.savefig(ppp + "EEG1.jpg")

    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    df_y = df_signals["EEG2"]

    ax1.plot(df_y)
    ax1.set_ylabel('µV')
    ax1.set_title('Amplitude')
    ax1.set_xticklabels(['', '', '', '', '', ''])

    vmin = 20 * np.log10(np.max(df_y)) - 58.1  # hide anything below -40 dBc
    ax2.specgram(df_y, Fs=500, cmap=cm.jet, vmin=vmin);
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('Freq (Hz)');
    ax2.set_xlabel('Time (sec)');
    ax2.set_title('Spectrogram');


    fig1.savefig(ppp + "EEG2.jpg")
    my_zip.write(ppp + "EEG1.jpg")
    my_zip.write(ppp + "EEG2.jpg")
    plt.close()
    my_zip.close()

    # Load the model back
    loaded_model = XGBClassifier()
    loaded_model.load_model('Mice.model')

    # Now you can use the loaded_model for predictions
    predictions = loaded_model.predict(df_data)

    result = sum(predictions) / len(predictions)

    if result>=0.5:
        final_pred="Outlier"
    if result<0.5:
        final_pred="Normal"
    os.remove("static/uploads/" + filename)
    return render_template('EPC_result.html',prediction_text=str(final_pred) )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=8000, threaded=True)
