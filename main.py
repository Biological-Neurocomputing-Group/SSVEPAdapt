import os, time, copy, cmath
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8,3)
#from more_itertools import flatten

from scipy.signal import detrend
#import peakutils as peakutils
from scipy.signal import butter, lfilter, lfilter_zi


from sklearn.cross_decomposition import CCA
import scipy.stats

NUM_SENSORS = 16+1
SAMPLERATE = 512;
NUM_TRAIN = 5
FREQ_STIMULUS_LIST = np.arange(8, 15+1)#last one is not included
NUM_HARMONICS = 3
HARMONICS_LIST = np.arange(1, NUM_HARMONICS+2)#last one is not included
MIN_FREQ = 6
MAX_FREQ = 65
file_path = ''

class SignalManager(object):

    def process_EEG_signal(data, freq_sampling = SAMPLERATE):
        sensor_size, data_size = data.shape

        # removing first data to avoid noise
        # data = np.stack(data, axis=0)
        # data = data[:, SAMPLERATE:]
        data = data.T
        data = data[freq_sampling - 1:, :-1]
        data = data.T

        return SignalManager.filter_signal_FFT(data, freq_sampling, MIN_FREQ, MAX_FREQ)

    def filter_signal_FFT(data, sampledata, min_freq, max_freq):

        output_data = []

        fs = sampledata
        n = max(data.shape) #3073
        t = 1 / fs
        time = np.linspace(0, t * n, n)
        interval_time = time[1] - time[0]

        for original_signal in data:

            f_signal = np.fft.fft(original_signal)
            W = np.fft.fftfreq(original_signal.size, d=interval_time)

            cut_f_signal = copy.deepcopy(f_signal)#f_signal.copy()
            cut_f_signal[(abs(W) < min_freq)] = 0.0
            cut_f_signal[(abs(W) > max_freq)] = 0.0

            filtered_signal = np.fft.ifft(cut_f_signal)
            output_data.append(filtered_signal)

        # List of nparay to nparray of nparray
        #print(len(output_data))
        data_out = np.stack(output_data, axis=1)

        return data_out.real, interval_time

    def model_CCA(X, Y):

        #method 1
        #--------
        #cca_1 = CCA()
        #X_c, Y_c = cca_1.fit_transform(X, Y)

        #result = np.corrcoef(X_c, Y_c)[0, 1]
        #print('r =', result)

        #method 2
        #--------
        result=[]
        num_components = 1 #Y.shape[1]
        cca = CCA(num_components)

        """
        U, V = cca.fit_transform(X, Y)
        #Pearson correlation coefficient
        for i in range(num_components):
            #result.append(np.corrcoef(U[:, i], V[:, i])[0, 1])
            result.append(np.corrcoef(U.T, V.T)[0,1])
        """
        try:
            X_c, Y_c = cca.fit_transform(X, Y)
            #print(X.shape)
            #print(Y.shape)
            corrcoef, p_value = scipy.stats.pearsonr(X_c, Y_c)
            result = corrcoef
        except Exception as err:
            print(err)
            result=[0]
        
        return max(result)

    def analize_EEG_signal(data, interval_time):

        serie_t = np.arange(0.0,max(data.shape))*interval_time #

        coeff_list = []

        #for harm in harmonics:
        for freq in FREQ_STIMULUS_LIST:
            array_Y = None
            for harm in HARMONICS_LIST:
                sin_signal_serie = np.sin(freq * 2 * np.pi * serie_t * harm)
                cos_signal_serie = np.cos(freq * 2 * np.pi * serie_t * harm)

                subarray =  np.concatenate(([sin_signal_serie], [cos_signal_serie]))

                if array_Y is None:
                    array_Y = subarray
                else:
                    array_Y = np.concatenate(([array_Y, subarray]), axis=0)

            x = data
            y = array_Y.T

            #Removing last row to match size with Y
            #x= np.delete(x, 0, axis=0)

            #print(np.array(data).shape)
            #print(np.array(array_Y).shape)
            coeff_r = SignalManager.model_CCA(x, y)
            coeff_list.append(coeff_r)

        max_r = max(coeff_list)
        max_r_position = coeff_list.index(max_r)

        s_power = coeff_list[max_r_position]**2
        n_power=[]
        for x in coeff_list:
            if x != max_r:
                n_power.append(x)
        n_power = np.mean(np.array(n_power)**2)
        s_relative_power = s_power / n_power

        f_detected = FREQ_STIMULUS_LIST[max_r_position]

        return f_detected, s_relative_power

class SerialManager(object):
    #def __init__(self, config, filemanager):
    #    self.filemanager = filemanager
    #    self.ports = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2', '/dev/ttyUSB3', 'nothing']

    def read_matrix_LE(file_name):
        # little-endian single precision float
        data = np.fromfile(file_name, dtype='<f4')
        #Resize properly the readed data
        return data.reshape([NUM_SENSORS, len(data) // NUM_SENSORS], order='F')


class AppManager(object):
    def set_variables(self,path, samplerate, num_sensors, num_totaltimes, freq_min, freq_max, fulllist):
        global NUM_SENSORS, SAMPLERATE, NUM_TRAIN, FREQ_STIMULUS_LIST, file_path

        NUM_SENSORS = num_sensors + 1
        SAMPLERATE = samplerate
        NUM_TRAIN = num_totaltimes
        if len(fulllist) == 0:
            FREQ_STIMULUS_LIST = np.arange(freq_min, freq_max + 1)
        else:
            FREQ_STIMULUS_LIST = fulllist

        file_path = path
        print('Folder :', file_path)
        print('freq list:', FREQ_STIMULUS_LIST)
        print('frew pos.:', np.arange(0, len(FREQ_STIMULUS_LIST)))
        print('Harm list:', HARMONICS_LIST)
        print('-----------')

    def find_best_electrodes(self, best_freq_detected, best_freq_detected_SNR):
        print('-----------')
        print('Step 1: best electrodes')
        freq_pos = np.where(FREQ_STIMULUS_LIST == best_freq_detected)[0][0]
        print('Best freq.', FREQ_STIMULUS_LIST[freq_pos])

        ### Geting the average of SNR, now is a input parameters
        """avg_s_power = 0
        for trial in np.arange(0, NUM_TRAIN):
            filename = file_path + 'data_' + str(freq_pos + 1) + '_' + str(trial + 1) + '.bin'
            data = SerialManager.read_matrix_LE(filename)  # leer_gtec *.bin
            data, delta = SignalManager.process_EEG_signal(data)
            f_detected, s_relative_power = SignalManager.analize_EEG_signal(data, interval_time=delta)
            avg_s_power += s_relative_power
            #avg_s_power.append(s_relative_power)

        avg_s_power /= NUM_TRAIN
        #avg_s_power = min(avg_s_power)
        print('AVG SNR:', avg_s_power)
        """
        #print('AVG SNR:', best_freq_detected_SNR)
        ###

        sensor_list_total = list(np.arange(0, NUM_SENSORS -1))
        #print(sensor_list_total)


        list_bad_electrodes_pos = []
        #list_good_eletrodes_pos = []
        dict_elec_snr = {}

        sensor_pos = 0
        sensor_list = copy.deepcopy(sensor_list_total)
        while(sensor_pos <= sensor_list[-1]): #sensor_list[-1]

            list_bad_electrodes_pos=[]#only 1 sensor to remove and test its effects in the CCA process
            list_bad_electrodes_pos.append(sensor_pos)
            result_s_power = 0
            f_detected = 0
            for trial in list(np.arange(0, NUM_TRAIN)):

                try:
                    # filename = file_path + 'data_' + str(freq_pos + 1) + '_' + str(trial+1) + '.bin'
                    filename = AppManager.generate_filename(trial + 1, 'freq', FREQ_STIMULUS_LIST[freq_pos])

                    data = SerialManager.read_matrix_LE(filename)  # leer_gtec *.bin

                    # Removing electrodes list with poor SNR
                    bad_list = sorted(list_bad_electrodes_pos,reverse=True) #primero los indices mayores para no solapar elementos a borrar
                    data = np.delete(data, bad_list, axis=0)

                    #print(data.shape)
                    data, delta = SignalManager.process_EEG_signal(data)
                    f_detected, s_relative_power = SignalManager.analize_EEG_signal(data, interval_time=delta)
                    result_s_power += s_relative_power

                except Exception as e:
                    print(e)

            result_s_power /= NUM_TRAIN

            #Old funtionality
            """
            if result_s_power < best_freq_detected_SNR and f_detected != best_freq_detected:
                #Si baja La SNR significa que el electrodo que acabo de quitar era bueno -> se mantiene
                list_bad_electrodes_pos.remove(sensor_pos)
                list_good_eletrodes_pos.append(sensor_pos)
            else:
                sensor_list.remove(sensor_pos)  # removing element by index
            """

            #new funtionality, we use a diccionary
            dict_elec_snr[sensor_pos] = round(result_s_power, 4)
            sensor_pos +=1

        #sorted(set(list_bad_electrodes_pos))

        #print('Bad sensors pos:',list_bad_electrodes_pos)
        #print('Good sensors pos:',list_good_eletrodes_pos)

        # 0 for keys, 1 for values
        # reverse=False. los mas bajos primero, significa que el electrodo eliminado aportaba mucha informacion
        ordered_s_power = sorted(dict_elec_snr.items(), key=lambda x: x[1], reverse=False)
        #print(ordered_s_power)
        list_good_eletrodes_pos=[]
        list_good_eletrodes_snr = []
        for key, value in ordered_s_power:
            list_good_eletrodes_pos.append(key)
            list_good_eletrodes_snr.append(value)

        return list_good_eletrodes_pos, list_good_eletrodes_snr

    def generate_filename(time, concept, value):
        return os.path.join(file_path,'record_'+str(time)+'_'+ str(concept) +'_'+str(float(value))+'.bin')

    def find_best_freq(self):#,file_path, samplerate, num_sensors, num_totaltimes, freq_min, freq_max):

        print('Processing files...')
        start = time.time()
        result_right = {}
        result_s_power = {}

        for freq_pos in np.arange(0, len(FREQ_STIMULUS_LIST)):
            result_right[freq_pos] = 0
            result_s_power[freq_pos] = 0

            for trial in np.arange(0, NUM_TRAIN):
                try:
                    #old system
                    #filename = file_path + 'data_' + str(freq_pos + 1) + '_' + str(trial + 1) + '.bin'
                    #new system

                    filename = AppManager.generate_filename(trial+1, 'freq', float(FREQ_STIMULUS_LIST[freq_pos]))
                    print(filename)

                    data = SerialManager.read_matrix_LE(filename)  # leer_gtec *.bin
                    data, delta = SignalManager.process_EEG_signal(data)
                    f_detected, s_relative_power = SignalManager.analize_EEG_signal(data, interval_time=delta)

                    #plot_example(data,15)
                    f_expected = FREQ_STIMULUS_LIST[freq_pos]
                    harm_high = f_expected * HARMONICS_LIST
                    harm_low = f_expected / HARMONICS_LIST

                    if f_detected in harm_high or f_detected in harm_low:
                        result_right[freq_pos] += 1

                    result_s_power[freq_pos] += s_relative_power
                except Exception as e:
                    print(e)

            result_s_power[freq_pos] /= NUM_TRAIN

        print('------')

        uniqueValues = sorted(set(result_right.values()), reverse=True)
        finalist_freq = []
        finalist_snr = []
        for u_value in uniqueValues:
            sublist = [k for k, v in result_right.items() if v == u_value]
            sub2_list = {}
            for key in sublist:
                sub2_list[key] = result_s_power[key]

            # 0 for keys, 1 for values
            ordered_s_power = sorted(sub2_list.items(), key=lambda x: x[1], reverse=True)
            for key, value in ordered_s_power:
                print('freq:', FREQ_STIMULUS_LIST[key], ', hits:', result_right[key], ', RCC:', round(value, 4))
                finalist_freq.append(FREQ_STIMULUS_LIST[key])
                finalist_snr.append(round(value, 4))

        end = time.time()
        print(round(end - start, 2), 'seconds of execution')

        return finalist_freq, finalist_snr


    def find_best_4_freq(self, good_sensor_list):
        print('Processing files...')
        start = time.time()
        result_right = {}
        result_s_power = {}

        for freq_pos in np.arange(0, len(FREQ_STIMULUS_LIST)):
            result_right[freq_pos] = 0
            result_s_power[freq_pos] = 0

            for trial in np.arange(0, NUM_TRAIN):
                try:
                    # old system
                    # filename = file_path + 'data_' + str(freq_pos + 1) + '_' + str(trial + 1) + '.bin'
                    # new system

                    filename = AppManager.generate_filename(trial + 1, 'online', FREQ_STIMULUS_LIST[freq_pos])
                    print(filename)
                    try:
                        data = SerialManager.read_matrix_LE(filename)  # leer_gtec *.bin
                    except Exception as e:
                        continue

                    # Removing electrodes list with poor SNR
                    if (len(good_sensor_list) > 0):
                        bad_sensors = []
                        sensor_list_total = list(np.arange(0, NUM_SENSORS-1))
                        for x in sensor_list_total:
                            if x not in good_sensor_list:
                                bad_sensors.append(x)
                        # print('bad sensors',bad_sensors)
                        #print(data.shape)
                        if (len(bad_sensors) > 0):
                            data = np.delete(data, bad_sensors, axis=0)
                        #print(data.shape)

                    data, delta = SignalManager.process_EEG_signal(data)
                    f_detected, s_relative_power = SignalManager.analize_EEG_signal(data, interval_time=delta)

                    f_expected = FREQ_STIMULUS_LIST[freq_pos]
                    harm_high = f_expected * HARMONICS_LIST
                    harm_low = f_expected / HARMONICS_LIST

                    if f_detected in harm_high or f_detected in harm_low:
                        result_right[freq_pos] += 1

                    result_s_power[freq_pos] += s_relative_power
                except Exception as e:
                    print(e)

            result_s_power[freq_pos] /= NUM_TRAIN

        print('------')

        uniqueValues = sorted(set(result_right.values()), reverse=True)
        finalist_freq = []
        finalist_snr = []
        for u_value in uniqueValues:
            sublist = [k for k, v in result_right.items() if v == u_value]
            sub2_list = {}
            for key in sublist:
                sub2_list[key] = result_s_power[key]

            # 0 for keys, 1 for values
            ordered_s_power = sorted(sub2_list.items(), key=lambda x: x[1], reverse=True)
            for key, value in ordered_s_power:
                print('freq:', FREQ_STIMULUS_LIST[key], ', hits:', result_right[key], ', RCC:', round(value, 4))
                finalist_freq.append(FREQ_STIMULUS_LIST[key])
                finalist_snr.append(round(value, 4))

        end = time.time()
        print(round(end - start, 2), 'seconds of execution')


        score_best_4_freq=0
        for pos in range(0,4):
            print(pos, finalist_freq[pos], result_right[pos])
            score_best_4_freq += result_right[pos]

        maximum_score_4_freq = NUM_TRAIN * 4
        total_score = score_best_4_freq / maximum_score_4_freq * 100

        return finalist_freq, finalist_snr, total_score

    def find_best_phase(self, best_freq, phase_list, phase_value, times_pi):
        list_freq_phase = []
        value = 0
        if best_freq in phase_list:
            pos = phase_list.index(best_freq)
            value = phase_value[pos]

        for x in range(0, times_pi):
            # Setting like that: 0 0.5 1.5 0. 0.5 ...
            list_freq_phase.append(value)
            if x >= 0: value += 0.5
            if value > 1.5: value = 0.0

        return list_freq_phase



"""
#######

#Simple program
#--------------
start = time.time()
data= SerialManager.read_matrix_LE(FILENAME) #leer_gtec
 
data, delta = SignalManager.process_EEG_signal(data)
coeff_list = SignalManager.analize_EEG_signal(data, interval_time=delta)
max_position = coeff_list.index(max(coeff_list))
#"""

#Complete program
#--------------
"""
Old system
file_path = '/Users/Aaron/MEGAsync/EEG_code_data/AaronPerez/SISTEMA_DE_GRABACION/Grabaciones/Aaron_16_05/bin/'
#file_path = '/Users/Aaron/MEGAsync/EEG_code_data/AaronPerez/SISTEMA_DE_GRABACION/Grabaciones/junio4_juan1/bin/'
#file_path = '/Users/Aaron/MEGAsync/EEG_code_data/AaronPerez/SISTEMA_DE_GRABACION/Grabaciones/junio5_sergio/bin/'
"""
"""
samplerate=512
num_sensors=16
num_totaltimes=3#5
freq_min=8
freq_max=15
path='/Users/Aaron/Desktop/records/190731_1236_Roy_freq_pha/' #''/home/gnb/Escritorio/TFM_Aaron/'

app = AppManager()
app.set_variables(path,samplerate,num_sensors,num_totaltimes,freq_min,freq_max,[])

print('Step 1: best freq and electrodes')
f_list, f_list_snr = app.find_best_freq() #output list freq:SNR
best_f = f_list[0]
best_f_snr = f_list_snr[0]
print('best frequency:',best_f, 'SNR:',best_f_snr)

sorted_electodes, sorted_SNR = app.find_best_electrodes(best_f, best_f_snr) #Poner la mejor frecuencia

print('Freqs:',f_list[0],f_list[1],f_list[2],f_list[3])
print('Elec.:',sorted_electodes)
print('SNR.:',sorted_SNR)
"""

"""


    #por cada freq
        #por cada fase
            #leer fichero
            #CCA
            #f == f_detected? Yes OK


#PREGUNTAR A PABLO
# Si el usuario hace una prueba con poco resultado, los bin?
# Usar posiciones en ficheros


# Validation: check best 4
# Resultado, las 4 mejores frecuencias deberia ser las detectadas
# Si no permitir elegir otras
" ""
default_phase_list  = [8, 9, 10, 11, 12, 13, 14, 15]
default_phase_value = [0.0, 0.5, 1.0, 1.5, 0.0, 0.5, 1.0, 1.5]

lista = find_best_phase(9,default_phase_list, default_phase_value,4)
print(lista)
"""

def plot_allelectrodes_onetrial(data, num_sensor):
    T = 1 / SAMPLERATE;
    t = np.arange(0, len(data[num_sensor])) * T
    # chunk = data[1][0:17]

    #for i in range(0, NUM_SENSORS):
    #    plt.plot(t, data[i])
    t = np.arange(0, len(data[num_sensor])) * T
    plt.plot(data[2])
    plt.show()

def plot_signal(pos,serie1,serie2,label,ylabel, xlabel):
    plt.grid(color='grey', linestyle='-', linewidth=0.5)
    plt.rcParams["figure.figsize"] = [10.0, 5.0]
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if(pos >0):
        plt.subplot(pos)
    plt.title(label)
    plt.plot(serie1,serie2)



def plot_sine_Arduino(freq1, freq2):
    fs = 512  # sampling rate, Hz, must be integer
    duration1 = 1.0  # in seconds, may be float
    duration2 = 2.0  # in seconds, may be float

    #time = np.arange(fs * duration)

    signal_1_1 = np.sin(2 * np.pi * np.arange(fs * duration1) * freq1 / fs * 0)
    signal_1_2 = np.sin(2 * np.pi * np.arange(fs * duration2) * freq1 / fs)

    ## Movemos la fase de pi a pi/2 es decir 90%
    signal_fft = np.fft.rfft(signal_1_2)
    signal_1_2_fft = signal_fft * cmath.rect(1., np.pi / 2)
    signal_1_2 = np.fft.irfft(signal_1_2_fft)

    singal_final_1 = np.concatenate((signal_1_1, signal_1_2), axis=None)



    signal_2_1 = np.sin(2 * np.pi * np.arange(fs * duration1) * freq2 / fs * 0)
    signal_2_2 = np.sin(2 * np.pi * np.arange(fs * duration2) * freq2 / fs)
    singal_final_2 = np.concatenate((signal_2_1, signal_2_2), axis=None)

    n = len(singal_final_1)
    t = 1 / fs
    time = np.linspace(0, t * n, n)
    plt.xlim([0, 2])


    plt.rcParams.update({'font.size': 14})
    axis_font = {'size': '14'}

    plt.title('Señales de estimulación LED: '+str(freq1)+'Hz a 0.5pi y '+str(freq2)+' HZ a 0pi', **axis_font)
    plt.ylabel("Amplitude", **axis_font)
    plt.xlabel("Time [s]", **axis_font)

    plt.plot(time, singal_final_1, 'b', label = '1 seno a '+str(freq1)+' Hz 0.5pi',linewidth=0.8)
    plt.plot(time, singal_final_2, color='orange', label ='1 seno a ' + str(freq2) + ' Hz 0pi', linewidth=0.8)
    plt.legend(loc='upper left')
    plt.grid()

    plt.savefig('Signal_Arduino_' + str(int(freq1)) +'_y_'+ str(int(freq2)) +'_Hz.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_signal_EEG_simple(trial,freq):
    FILENAME = AppManager.generate_filename(trial, 'freq', freq)
    print(FILENAME)

    data = SerialManager.read_matrix_LE(FILENAME)  # leer_gtec *.bin
    data, delta = SignalManager.process_EEG_signal(data)
    #f_detected, s_relative_power = SignalManager.analize_EEG_signal(data, interval_time=delta)
    data = data.T

    sensor_list, size = data.shape
    n = size
    t = 1 / SAMPLERATE
    time = np.linspace(0, t * n, n)

    chunk_time = time[:size // 2]
    chunk_data = data[6][:size//2]
    plt.xlim([0, 3])

    plt.rcParams.update({'font.size': 14})
    axis_font = {'size': '14'}

    plt.title('Señal de EEG registrada en el electrodo OZ.', **axis_font)
    plt.ylabel("Amplitude", **axis_font)
    plt.xlabel("Time [s]", **axis_font)


    plt.plot(chunk_time, chunk_data,'b',label='Señal EEG Electrodo OZ' ,linewidth=0.8)
    plt.axvline(x=1, color='red', alpha=1, linewidth=1.0)
    plt.legend(loc='upper right')

    plt.grid()
    plt.savefig('Signal_EEG_simple.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_time_serie_Electrodes(trial,freq):
    FILENAME = AppManager.generate_filename(trial, 'freq', freq)
    print(FILENAME)

    data = SerialManager.read_matrix_LE(FILENAME)  # leer_gtec *.bin
    data, delta = SignalManager.process_EEG_signal(data)

    data = data.T
    sensor_list, size = data.shape
    n=size
    t = 1 / SAMPLERATE
    time = np.linspace(0,t*n,n)
    p = time[1] - time[0]

    #FFT processing
    fft_signal = np.fft.fft(data[6])
    sampling_interval = time[1]-time[0]
    N = len(fft_signal)
    frequencies = np.linspace(0, 1/sampling_interval, N)

    #mitad_freq = frequencies[:N//2]
    #mitad_data = abs(data[6])[:N//2]*1/N
    #mitad_data = data_01_P03[:N//2] #*1/N

   # plt.axes.set_xlim([xmin,xmax])

    plt.ylim([-60,60])
    serie1 = data[5][:size // 2]
    serie2 = data[6][:size // 2]
    time = time[:size // 2]

    serie_diff = serie1 - serie2

    serie1 -= 40 #'Electrodo O1'
    serie2 += 40 #'Electrodo OZ'


    #Las posiciones de los electrodos empiezan en 0, si PZ es 7 en la realidad, en el array será 6
    plt.plot(time, serie1,'b', linewidth=0.8, label='Electrodo O1')
    plt.plot(time, serie2, color='orange',linewidth=0.8, label='Electrodo OZ')
    plt.plot(time, serie_diff,'g', linewidth=0.8, label='O1 - OZ')
    plt.axvline(x=1, color='red', alpha=1, linewidth=1.0)

    plt.rcParams.update({'font.size': 14})
    axis_font = {'size': '14'}

    plt.title('Señales de EEG registrada en O1 y OZ, '+str(freq)+'Hz fase 0', **axis_font)
    plt.ylabel("Amplitud", **axis_font)
    plt.xlabel("Tiempo [s]", **axis_font)
    plt.legend(loc='upper right', )
    plt.grid()
    plt.savefig('Signal_EEG_electrodes.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

def plot_signal_Arduino_EEG(trial,freq):
    FILENAME = AppManager.generate_filename(trial, 'freq', freq)
    print(FILENAME)

    #Serie EEG
    data = SerialManager.read_matrix_LE(FILENAME)  # leer_gtec *.bin
    data, delta = SignalManager.process_EEG_signal(data)
    data = data.T

    sensor_list, size = data.shape
    n = size
    t = 1 / SAMPLERATE
    time = np.linspace(0, t * n, n)

    chunk_time = time[:size // 2]
    serie1 = data[5][:size // 2] # O1
    serie2 = data[6][:size // 2] # Oz

    serie_diff = serie1 - serie2
    chunk_data_EEG = serie_diff[:size // 2]


    #Serie Arduino
    fs = 512  # sampling rate, Hz, must be integer
    duration1 = 1.0  # in seconds, may be float
    duration2 = 2.0  # in seconds, may be float
    f = freq  # sine frequency, Hz, may be float

    signal1 = np.sin(2 * np.pi * np.arange(fs * duration1) * f / fs * 0)
    signal2 = np.sin(2 * np.pi * np.arange(fs * duration2) * f / fs)
    singal_final = np.concatenate((signal1, signal2), axis=None)

    chunk_Arduino = singal_final[:size // 2]

    plt.xlim([0.9, 1.6])

    # Normalizing data
    chunk_data_EEG = 2 * (chunk_data_EEG - min(chunk_data_EEG)) / (max(chunk_data_EEG) - min(chunk_data_EEG)) -1

    plt.plot(chunk_time, chunk_Arduino, 'b',linewidth=0.8, label='Señal Arduino')
    plt.plot(chunk_time,chunk_data_EEG,color='orange',linewidth=0.8, label='Señal EEG O1 - OZ')
    plt.axvline(x=1, color='red', alpha=1, linewidth=1.0)

    plt.rcParams.update({'font.size': 14})
    axis_font = {'size': '14'}

    plt.title('Señales de Arduino y EEG O1 - OZ, ' + str(freq) + 'Hz fase 0', **axis_font)
    plt.ylabel("Amplitud", **axis_font)
    plt.xlabel("Tiempo [s]", **axis_font)
    plt.legend(loc='upper right', )
    plt.grid()

    plt.savefig('Signal_Arduino_EEG_' + str(freq) + '_Hz.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

def plot_fft_serie(trial,freq):
    FILENAME = AppManager.generate_filename(trial, 'freq', freq)
    print(FILENAME)

    data = SerialManager.read_matrix_LE(FILENAME)  # leer_gtec *.bin
    data, delta = SignalManager.process_EEG_signal(data)
    f_detected, s_relative_power = SignalManager.analize_EEG_signal(data, interval_time=delta)
    data = data.T

    sensor_list, size = data.shape
    n = size
    t = 1 / SAMPLERATE
    time = np.linspace(0, t * n, n)
    sampling_interval = time[1] - time[0]

    fft_signal = np.fft.fft(data[6])

    N = len(fft_signal)
    frequencies = np.linspace(0, 1 / sampling_interval, N)

    mitad_freq = frequencies[:N // 7]
    mitad_data = abs(fft_signal)[:N // 7] * 1 / N

    plt.xlim([6, 18])

    plt.axvline(x=freq, color='red', alpha=1, linewidth=1.0)
    plt.plot(mitad_freq, mitad_data, 'b', label='F. est.= ' + str(freq) + 'Hz', linewidth=0.8)
    plt.rcParams.update({'font.size': 14})
    axis_font = {'size': '14'}

    plt.title('Frecuencias en señal EEG registrada en OZ, '+str(freq)+'Hz fase 0', **axis_font)
    plt.ylabel("Amplitud", **axis_font)
    plt.xlabel("Frecuencia [Hz]", **axis_font)
    plt.legend(loc='upper right', )
    plt.grid()

    plt.savefig('Signal_FFT_' + str(freq) + '_Hz.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

def plot_fft_CCA_singleStimulus(trial,freq):
    FILENAME = AppManager.generate_filename(trial, 'freq', freq)
    print(FILENAME)

    data = SerialManager.read_matrix_LE(FILENAME)  # leer_gtec *.bin
    data, delta = SignalManager.process_EEG_signal(data)
    f_detected, s_relative_power = SignalManager.analize_EEG_signal(data, interval_time=delta)
    data = data.T

    sensor_list, size = data.shape
    n=size
    t = 1 / SAMPLERATE
    time = np.linspace(0,t*n,n)
    sampling_interval = time[1] - time[0]


    fft_signal = np.fft.fft(data[6])

    N = len(fft_signal)
    frequencies = np.linspace(0, 1/sampling_interval, N)

    mitad_freq = frequencies[:N//7]
    mitad_data = abs(fft_signal)[:N//7]*1/N

    plt.xlim([6, 18])

    plt.plot(mitad_freq, mitad_data,'b', label='F. est.= ' + str(freq) + 'Hz',linewidth=0.8)
    plt.axvspan(f_detected-1, f_detected +1, facecolor='orange', alpha=0.3)

    plt.axvspan(f_detected-0.05, f_detected+0.05, facecolor='red', alpha=0.8,label='F. CCA = ' + str(f_detected) + 'Hz')

    plt.rcParams.update({'font.size': 14})
    axis_font = {'size': '14'}

    plt.title('Registro EEG con estímulo LED a ' + str(freq) + 'Hz', **axis_font)
    plt.ylabel("Amplitud",**axis_font)
    plt.xlabel("Frecuencia [Hz]",**axis_font)
    plt.legend(loc='upper right',)
    plt.grid()

    plt.savefig('SSVEP_'+str(freq)+'_Hz.png', dpi=300, bbox_inches='tight',pad_inches = 0)
    plt.show()

def plot_fft_CCA_fourStimulus(trial,freq):
    FILENAME = AppManager.generate_filename(trial, 'online', freq)
    print(FILENAME)

    data = SerialManager.read_matrix_LE(FILENAME)  # leer_gtec *.bin
    data, delta = SignalManager.process_EEG_signal(data)
    f_detected, s_relative_power = SignalManager.analize_EEG_signal(data, interval_time=delta)
    data = data.T

    sensor_list, size = data.shape
    n=size
    t = 1 / SAMPLERATE
    time = np.linspace(0,t*n,n)
    sampling_interval = time[1] - time[0]


    fft_signal = np.fft.fft(data[6])

    N = len(fft_signal)
    frequencies = np.linspace(0, 1/sampling_interval, N)

    mitad_freq = frequencies[:N//7]
    mitad_data = abs(fft_signal)[:N//7]*1/N

    plt.xlim([6, 18])

    plt.plot(mitad_freq, mitad_data,'b', label='F. est.= ' + str(freq) + 'Hz',linewidth=0.8)
    plt.axvspan(f_detected-1, f_detected +1, facecolor='orange', alpha=0.3)

    plt.axvspan(f_detected-0.05, f_detected+0.05, facecolor='red', alpha=0.8,label='F. CCA = ' + str(f_detected) + 'Hz')

    plt.rcParams.update({'font.size': 14})
    axis_font = {'size': '14'}

    plt.title('Registro EEG con estímulo LED a ' + str(freq) + 'Hz', **axis_font)
    plt.ylabel("Amplitud",**axis_font)
    plt.xlabel("Frecuencia [Hz]",**axis_font)
    plt.legend(loc='upper right',)
    plt.grid()

    plt.savefig('SSVEP_online'+str(freq)+'_Hz.png', dpi=300, bbox_inches='tight',pad_inches = 0)
    plt.show()

## Realización de gráficas.

#Variable global
file_path = '/Users/Aaron/Desktop/records/190731_1318_JuanRico_freq_pha/'

trials_to_read = 1
stimilus_freq = 15

#Serie temporal seno artificial del Arduino. OK
#plot_sine_Arduino(8,stimilus_freq)

#Serie temporal de 2 electrodos de EEG. OK
#plot_signal_EEG_simple(trials_to_read, stimilus_freq)

#Serie temporal de 2 electrodos de EEG. OK
#plot_signal_Arduino_EEG(trials_to_read, stimilus_freq)

#Serie tempral restando electdodos, para eliminar ruido. OK
#plot_time_serie_Electrodes(trials_to_read,stimilus_freq)

#Serie FFT
#plot_fft_serie(trials_to_read, stimilus_freq)

#Serie de FFT con CCA detectado, 1 estimuloa la vez . OK
#plot_fft_CCA_singleStimulus(trials_to_read, stimilus_freq)

#Serie de FFT con CCA detectado, 4 estímulos a la vez
#plot_fft_CCA_fourStimulus(trials_to_read, stimilus_freq)

