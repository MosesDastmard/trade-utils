
import numpy as np
from tqdm.notebook import tqdm
from scipy import stats

def Rsi(arr, window=14):
    deltas = np.diff(arr)
    seed = deltas[:window+1]
    up = seed[seed >= 0].sum()/window
    down = -seed[seed < 0].sum()/window
    rs = up/down
    rsi = np.zeros_like(arr)
    rsi[:window] = 100. - 100./(1.+rs)
    for i in range(window, len(arr)):
        delta = deltas[i-1]  # cause the diff is 1 shorter
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(window-1) + upval)/window
        down = (down*(window-1) + downval)/window
        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)
    return rsi


def Switch(a, b):
    c = np.sign(b-a)
    c[c == 0] = 1
    d = np.pad(np.abs(np.diff(c)), (1,0))
    return c*d/2

def HIGH(arr):
    if (arr.size%2 == 0):
        arr = arr[1:]
    m = arr.size//2
    return arr.max() == arr[m]

def LOW(arr):
    if (arr.size%2 == 0):
        arr = arr[1:]
    m = arr.size//2
    return arr.min() == arr[m]


def getLabel(df, window_size = 30, order = -1):
    label = []
    cof = np.array([(window_size - j)**order for j in range(window_size)])
    cof = cof/cof.sum()*10000
    close = df['close'].values
    for i, price in tqdm(enumerate(df['close']), total = len(df)):
        if i > (len(df['close']) - window_size):
            target = 0
        else:
            chunk = close[i:(i + window_size)]
            open_price = chunk[0]
            close_price = chunk[-1]
            dif = open_price - chunk
            if order == -1:
                target = (close_price - open_price)*10000
            else:
                target = (dif*cof).sum()
        label.append(target)
    return np.array(label)

def string2min(s):
    h,m = str(s).split(":")
#     print(h)
#     print(m)
    return int(h[2:])*60 + int(m[0:-2])

def getMinute(df):
    return df[['time']].apply(string2min, axis=1, raw=True)

def Strider(arr, window=14, stride=1):  
    arrPaded = np.concatenate([np.empty(window-1) + np.NaN, arr])
    nrows = ((arrPaded.size-window)//stride)+1
    n = arrPaded.strides[0]
    return np.lib.stride_tricks.as_strided(arrPaded, shape=(nrows,window), strides=(stride*n,n))

def Slider(arr, fun, window=14, stride=1):
    arrStrided = Strider(arr, window=window, stride=stride)
    return np.apply_along_axis(fun, axis=1, arr=arrStrided)

def Compare(arrA, arrB, fun, window=14, stride=1):
    arrAStrided = Strider(arrA, window=window, stride=stride)
    arrBStrided = Strider(arrB, window=window, stride=stride)
    r = list()
    for i in range(arrAStrided.shape[0]):
        r.append(fun(arrAStrided[i,:], arrBStrided[i,:]))
    return np.array(r)

def Ema(arr, window):
    """
    returns an n period exponential moving average for
    the time series s

    s is a list ordered from oldest (index 0) to most
    recent (index -1)
    n is an integer

    returns a numeric array of the exponential
    moving average
    """
    ema = []
    j = 1

    #get n sma first and calculate the next n period ema
    sma = sum(arr[:window]) / window
    multiplier = 2 / float(1 + window)
    ema.append(sma)

    #EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
    ema.append(( (arr[window] - sma) * multiplier) + sma)

    #now calculate the rest of the values
    for i in arr[window+1:]:
        tmp = ( (i - ema[j]) * multiplier) + ema[j]
        j = j + 1
        ema.append(tmp)

    return np.concatenate([np.empty(window-1) + np.NaN, np.array(ema)])
def Enter(arrA, arrB):
    return ((arrA[0] > arrB[0]) & (arrA[1] < arrB[1]))
    



def Slope(arr):
    slope, _, _, _, _=  stats.linregress(x = arr, y = np.array(range(arr.size)))
