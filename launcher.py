#! /usr/bin/env python
######################
#
# Tecnologías para la adaptación al usuario 
# en interfaces cerebro-máquina basados 
# en potenciales visuales evocados.
# 
# Aarón Pérez Martín - aapemar@gmail.com
# Universidad Autónoma de Madrid - 2019
#
# CC BY-NC-SA 4.0 License. 
# http://creativecommons.org/licenses/by-nc-sa/4.0/
#
######################

import sys, serial, os, shutil, subprocess, json, time, copy
from time import gmtime, strftime
from functools import partial
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import screen_main as MyScreen
from time import sleep
from datetime import datetime
from multiprocessing.pool import ThreadPool

from PyQt5.QtGui import QFont
from main import *
pool = ThreadPool(processes=1)

lbl_counter = ' trials'
lbl_percent = ' %'

#VARIABLES
#===========
generalFolder = ''
generalSubFolder = 'records'
gUSBamp_folder = 'gUSBamp_files'
gUSBamp_filename = 'data_master.bin'
gUSBamp_scriptname = 'gUSBampAPI_Demo'
sessionFolder = ''
session_cod = ''
file_path = ''
default_phase_list  = [8, 9, 10, 11, 12, 13, 14, 15]
default_phase_value = [0.0, 0,5, 1.0, 1.5, 0.0, 0.5, 1.0, 1.5]

list_freq_to_test = []
list_freq_phase = []
current_freq_pos = 0

list_phas_to_test = []

#CONSTANTS
#===========
ARDUINO_PORT = '/dev/ttyACM0'
ARDUINO_SPEED = 9600

map_elect = ['Standard EEG map 10-20','Other']
genders = ['Male', 'Female']
best_elect_list = []

BEST_ELEC_TIMES  = 3#5
BEST_FREQ_TIMES  = 3#5
BEST_PHASE_TIMES = 1
BEST_ONLINE_TIMES= 3

BEST_FREQ_FROM = 8
BEST_FREQ_TO = 15
best_freq_list = []
#AUTO SELECTION OF BEST ELECTRODES

BEST_PHASE_FREQ = 8
BEST_PHASE_FROM = 0
BEST_PHASE_TO = 1.5
best_phase_list = []

map_online =['Online Test Applicaton','Other']

WAIT_SECONDS = 1
STATUSBAR_TIME = 1000
#============

myScreen = MyScreen.Ui_MainWindow() #create a new instance of the Ui you just cerated.

working_any_thread = False
elec_times = elec_times_total = freq_times = freq_times_total = phase_times = phase_times_total = online_times = online_times_total = 0

##____Folders____
def setWorkingFolder():
    global generalFolder, sessionFolder, session_cod
    refresh_window()
    generalFolder = os.getcwd()
    generalFolder = os.path.join(generalFolder, generalSubFolder)

    #user_name = myScreen.txt_username.text()
    #session_name = myScreen.txt_username.text()
    session_cod = myScreen.txt_session_cod.text()

    if (len(generalFolder) > 0 and len(session_cod)>0):
        if not os.path.exists(generalFolder):
            os.mkdir(generalFolder)

        sessionFolder = os.path.join(generalFolder,session_cod)
        if (len(sessionFolder) > 0):
            if not os.path.exists(sessionFolder):
                os.mkdir(sessionFolder)

            write_statusbar('Session folder ready')
    else:
        write_statusbar('No found session folder')
        refresh_window()

def moving_generatedfile_to(session_cod, newfilename):
    origin = os.path.join(os.getcwd(), gUSBamp_filename) #gUSBamp_folder,
    #origin = gUSBamp_filename
    destination = newfilename #os.path.join(generalFolder, session_cod, newfilename)
    #print(origin)
    print(destination)

    try: shutil.move(origin,destination)
    except Exception as e:
        print('ERROR: moving_file_to',str(e))

def export_to_json():
    refresh_window()
    if (len(generalFolder) > 0 and len(sessionFolder) > 0):

        student_data = {"user": []}
        # create a list
        data_holder = student_data["user"]
        data_holder.append({'username': myScreen.txt_username.text()})
        data_holder.append({'gender': myScreen.comboBox_gender.currentText()})
        data_holder.append({'age': myScreen.spin_age.value()})
        data_holder.append({'glasses': myScreen.checkBox_glasses.isChecked()})
        data_holder.append({'sessio_name': myScreen.txt_session_name.text()})
        data_holder.append({'sesion_cod': myScreen.txt_session_cod.text()})
        data_holder.append({'comments': myScreen.txt_comments.text()})

        data_holder.append({'best_elec': myScreen.txt_result_b_elec.text()})
        data_holder.append({'best_freq': myScreen.txt_result_b_freq.text()})
        data_holder.append({'times_elec_and_freq': freq_times})

        data_holder.append({'best_pha': myScreen.txt_result_b_phase.text()})
        data_holder.append({'times_pha': phase_times})

        data_holder.append({'best_online_freq': myScreen.txt_validation_result_b_elec.text()})
        data_holder.append({'times_online': online_times})
        data_holder.append({'online_score': myScreen.lbl_final_score.text()})

        file_path = os.path.join(sessionFolder,'data.json')
        with open(file_path, 'w') as outfile:
            json.dump(student_data, outfile)
        refresh_window()
        write_statusbar('Data exported!')
        outfile.close()
    else:
        write_statusbar('No found session folder')
        refresh_window()

##------------



##____Windows____
def compose_session_cod():
    global session_cod
    
    datetime_value = myScreen.session_datetime.dateTime().toString("yyMMdd_HHmm")

    newcod = datetime_value +'_'+ myScreen.txt_username.text() +'_'+ myScreen.txt_session_name.text()
    myScreen.txt_session_cod.setText(newcod)

    session_cod = myScreen.txt_session_cod.text()



def update_values():
    
    global elec_times_total,freq_times_total,phase_times_total
    global BEST_FREQ_FROM, BEST_FREQ_TO, BEST_PHASE_FREQ, BEST_PHASE_FROM, BEST_PHASE_TO

    #elec_times_total = int(myScreen.spin_times_elec.text())

    freq_times_total = int(myScreen.spin_times_freq.text())
    BEST_FREQ_FROM = float(myScreen.spin_freq_from.value())
    BEST_FREQ_TO =  float(myScreen.spin_freq_to.value())

    phase_times_total = int(myScreen.spin_times_pha.text())
    BEST_PHASE_FREQ = float(myScreen.spin_freq.value())
    BEST_PHASE_FROM = float(myScreen.spin_phase_from.value())
    BEST_PHASE_TO = float(myScreen.spin_phase_to.value())

def results_of_textbox():
    try:
        value = myScreen.txt_result_b_freq.text()
        if len(value) > 0:
            list = value.split(' ')
            myScreen.spin_freq.setValue(float(list[0]))
    except Exception as e:
        print(str(e))

def set_default_values_on_windows():
    myScreen.lbl_recording.setEnabled(False)
    myScreen.lbl_stimulus_monitor.setEnabled(False)
    myScreen.lbl_stimulus_led.setEnabled(True)

    myScreen.comboBox_gender.addItems(genders)
    myScreen.spin_age.setValue(18)
    myScreen.txt_username.textChanged.connect(compose_session_cod)
    myScreen.txt_session_name.textChanged.connect(compose_session_cod)
    myScreen.session_datetime.setCalendarPopup(True)

    myScreen.txt_result_b_freq.textChanged.connect(results_of_textbox)

    myScreen.txt_session_cod.textChanged.connect(compose_session_cod)

    now = QtCore.QDateTime.currentDateTime()
    myScreen.session_datetime.setDateTime(now)
    myScreen.session_datetime.dateTimeChanged.connect(compose_session_cod)

    myScreen.comboBox_elect.clear()
    myScreen.comboBox_elect.addItems(map_elect)
    myScreen.txt_usbport.setText(ARDUINO_PORT)

    myScreen.spin_freq_from.setValue(BEST_FREQ_FROM)
    myScreen.spin_freq_from.editingFinished.connect(update_values)
    myScreen.spin_freq_to.setValue(BEST_FREQ_TO)
    myScreen.spin_freq_to.editingFinished.connect(update_values)
    myScreen.spin_times_freq.setValue(BEST_FREQ_TIMES)
    myScreen.spin_times_freq.editingFinished.connect(update_values)

    myScreen.spin_freq.setValue(BEST_PHASE_FREQ)
    myScreen.spin_freq.editingFinished.connect(update_values)
    myScreen.spin_phase_from.setValue(BEST_PHASE_FROM)
    myScreen.spin_phase_from.editingFinished.connect(update_values)
    myScreen.spin_phase_to.setValue(BEST_PHASE_TO)
    myScreen.spin_phase_to.editingFinished.connect(update_values)
    myScreen.spin_times_pha.setValue(BEST_PHASE_TIMES)
    myScreen.spin_times_pha.editingFinished.connect(update_values)

    myScreen.spin_times_app.setValue(BEST_ONLINE_TIMES)
    myScreen.spin_times_app.editingFinished.connect(update_values)

    myScreen.comboBox_app.clear()
    myScreen.comboBox_app.addItems(map_online)

    myScreen.txt_other_electrodes.setEnabled(False)
    myScreen.txt_other_frequencies.setEnabled(False)
    myScreen.txt_other_phases.setEnabled(False)

def refresh_window():

    if(working_any_thread):
        myScreen.lbl_recording.setEnabled(True)
    else:
        myScreen.lbl_recording.setEnabled(False)
    app.processEvents()
    time.sleep(0.1)

def write_statusbar(txt):
    myScreen.statusbar.showMessage(txt)
    refresh_window()
##------------

##____Arduino____
def send_to_arduino_NO(parameters):
    print('EXECUTING:',parameters)
    device = serial.Serial()
    device.port = ARDUINO_PORT
    device.baudrate = ARDUINO_SPEED
    device.timeout = 0.2

    try:
        device.open()
        device.flushInput()
        device.write(parameters.encode())#parameters.encode("utf-8")
        sleep(2)
        #x = device.readline()
        #print('Receiving:',x)
        device.close()

    except Exception as e:
        print(str(e))


def set_arduino():
    execute('sudo stty -F '+ str(ARDUINO_PORT)+' cs8 9600 -ignbrk -brkint -icrnl -imaxbel -opost -onlcr -isig -icanon -iexten -echo -echoe -echok -echoctl -echoke noflsh -ixon -crtscts')
    execute('sudo chmod 777 '+ str(ARDUINO_PORT)+'')
    execute('sudo gnome-terminal -x bash -c "sudo cu -l '+ str(ARDUINO_PORT)+' -s 9600"')#; sleep 5;
    sleep(2)
##------------

##____External____
def execute(str_command, showresult=False):
    if(showresult):
        print('EXECUTING:',str_command)
    result = subprocess.run(str_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    sleep(0.25)
    err = result.stderr.decode("utf-8")
    out = result.stdout.decode("utf-8")
    if(showresult):
        print('Return code:',result.returncode)
        print('error:',err)
        print('output:',out)
    return err, out

def execute_CCA(time, concept, value, good_sensor_list, sensors_num):
    filename = generate_filename(time,concept,value)
    #print('NEW filename', filename)
    moving_generatedfile_to(session_cod,filename)
    #sleep(1)

    data = SerialManager.read_matrix_LE(os.path.join(generalFolder,session_cod,filename))

    # Removing electrodes list with poor SNR
    if(len(good_sensor_list)>0):
        bad_sensors = []
        sensor_list_total = list(np.arange(0, sensors_num))
        for x in sensor_list_total:
            if x not in good_sensor_list:
                bad_sensors.append(x)
        #print('bad sensors',bad_sensors)
        #print(data.shape)
        if(len(bad_sensors)>0):
            data = np.delete(data, bad_sensors, axis=0)
        #print(data.shape)


    data, delta = SignalManager.process_EEG_signal(data)
    f_detected, s_relative_power = SignalManager.analize_EEG_signal(data,interval_time=delta)
    #f_detected = 12
    print('trial:',time,'stimulus:',str(concept),value,', detected',str(concept),f_detected, '\n')

    return f_detected, s_relative_power

#-------------

def generate_filename(time, concept, value):
    newfilename = 'record_'+str(time)+'_'+ str(concept) +'_'+str(value)+'.bin'
    finalpath = os.path.join(generalFolder, session_cod, newfilename)
    return finalpath

def check_path():
    global generalFolder, generalFolder, file_path
    setWorkingFolder()
    if (len(myScreen.txt_username.text()) == 0 and len(myScreen.txt_session_name.text()) == 0):
        generalFolder = os.getcwd()
        #generalFolder = os.path.join(generalFolder, generalSubFolder)
        file_path = generalFolder
        print('a')
    else:
        compose_session_cod()

        file_path = sessionFolder
        print('b')
    print('SETTING FILE PATH:', file_path)

def Thread_step_2_best_elec(a, b):
    global freq_times, working_any_thread, list_freq_to_test, current_freq_pos, list_freq_phase
    new_freq_times = freq_times +1
    refresh_window()

    try:

        set_arduino()

        if new_freq_times <= myScreen.spin_times_freq.value(): #<=
            freq_times += 1
            if (len(list_freq_to_test) == 0):
                list_freq_to_test = np.arange(float(myScreen.spin_freq_from.value()),
                                              float(myScreen.spin_freq_to.value()) + 1)
                list_freq_phase=[]
                value = 0
                for x in range(0,len(list_freq_to_test)):
                    #Setting like that: 0 0.5 1.5 0. 0.5 ...
                    list_freq_phase.append(value)
                    if x>=0: value+= 0.5
                    if value >1.5: value=0.0

                print('Frec list',list_freq_to_test)
                print('Asoc Phase',list_freq_phase)

            list_freq_detected =[]
            list_s_power_freq = []

            current_freq_pos=0
            while(current_freq_pos < len(list_freq_to_test)):
                string = 'Freq: ' + str(list_freq_to_test[current_freq_pos]) + ', Phase: ' + str(
                    list_freq_phase[current_freq_pos])
                write_statusbar(string)
                #sleep(1)

                #Mode set on
                execute('sudo echo "179" > '+ str(ARDUINO_PORT)+'')

                #Configure Arduino
                f = str(list_freq_to_test[current_freq_pos])
                p = str(list_freq_phase[current_freq_pos])
                execute('sudo echo "'+f+' '+p+' 0.0 1.5 0.0 1.5 0.0 1.5" > '+ str(ARDUINO_PORT)+'')

                #gUSBamp trigger the led and record
                err,out = execute('sudo echo $$&')
                execute('sudo ./gUSBamp_files/gUSBampAPI_Demo '+str(out),showresult=False)

                #Turn off leds
                execute('sudo echo "999" > /dev/ttyACM0')


                f, s_power = execute_CCA(time=freq_times,concept='freq',value=list_freq_to_test[current_freq_pos], good_sensor_list=[], sensors_num=0)

                list_freq_detected.append(f)
                list_s_power_freq.append(round(s_power,4))

                current_freq_pos +=1

            print('freq detected:',list_freq_detected)
            print('RCC by freq  :',list_s_power_freq)

            write_statusbar('Step 2 has finished')
        else:

            samplerate=512
            num_sensors=16
            num_totaltimes=myScreen.spin_times_freq.value()
            freq_min=myScreen.spin_freq_from.value()
            freq_max=myScreen.spin_freq_to.value()

            #try:
            app = AppManager()
            app.set_variables(sessionFolder, samplerate, num_sensors, num_totaltimes, freq_min, freq_max,[])

            best_f_list, best_f_snr = app.find_best_freq()  # output list freq:SNR
            print('best frequency:', best_f_list[0], 'RCC:', best_f_snr[0])

            best_electodes, best_electodes_snr = app.find_best_electrodes(best_f_list[0], best_f_snr[0])  # Poner la mejor frecuencia

            # Step 2: best phase from best freq
            print('Step 3: Validation best 4 freq, elect and phase')
            print('Freqs:', best_f_list)
            print('Elec.:', best_electodes)
            print('Elec RCC.:', best_electodes_snr)

            string_f = ''
            for x in best_f_list:
                string_f += str(x) + ' '

            string_e = ''
            for x in best_electodes:
                string_e += str(x) + ' '

            myScreen.txt_result_b_freq.setText(string_f)
            myScreen.txt_result_b_elec.setText(string_e)

            write_statusbar('Times for step 2 is up!')

    except Exception as err:
        print(err)

    refresh_window()

    myScreen.lbl_count_b_freq.setText('')
    myScreen.lbl_count_b_freq.setText(str(freq_times) + lbl_counter)
    working_any_thread = False
    refresh_window()

def Thread_step_3_best_phase(a,b):
    global working_any_thread
    default_phase_list = [8, 9, 10, 11, 12, 13, 14, 15]
    default_phase_value = [0.0, 0.5, 1.0, 1.5, 0.0, 0.5, 1.0, 1.5]
    refresh_window()
    try:
        b_freq = float(myScreen.spin_freq.value())

        if b_freq > 0:
            samplerate = 512
            num_sensors = 16
            num_totaltimes = myScreen.spin_times_app.value()
            freq_min = min(default_phase_list) #[0]
            freq_max = max(default_phase_list) #[-1]

            app = AppManager()
            app.set_variables(sessionFolder, samplerate, num_sensors, num_totaltimes, freq_min, freq_max,[])

            phase_list = app.find_best_phase(b_freq, default_phase_list, default_phase_value, 4)
            print('Phase List:', phase_list)

            string_p = ''
            for x in phase_list[:4]:
                string_p += str(x) + ' '
            myScreen.txt_result_b_phase.setText(string_p)
        else:
            print("There is no enough data to continue")
            working_any_thread = False
            return

        myScreen.lbl_count_b_phase.setText('')
        myScreen.lbl_count_b_phase.setText(str(phase_times) + lbl_counter)

    except Exception as err:
        print(err)

    working_any_thread = False
    refresh_window()



def Thread_step_4_validation(a, b):
    global online_times, working_any_thread
    new_online_times = online_times + 1
    refresh_window()

    #list_freq_to_test = [8.0, 9.0, 10.0, 11.0]
    #list_freq_phase = [0.0, 0.5, 1.0, 1.5]
    #list_good_sensors_pos = [3, 4]

    #if myScreen.rd_app_other_freq.isChecked():

    try:
        #Frequencies
        f_list=[]
        if myScreen.rd_app_other_freq.isChecked():
            f_list = myScreen.txt_other_frequencies.text().split(' ')
        else:
            f_list = myScreen.txt_result_b_freq.text().split(' ')

        #f_list = f_list[:-1]
        list_freq_to_test = f_list[:4] #[8.0, 9.0, 10.0, 11.0]
        newArray=[]
        string_f = ''
        for x in list_freq_to_test:
            #if len(x) > 0 and x.isdigit():
            string_f += str(x) + ' '
            newArray.append(float(x))
        list_freq_to_test = copy.deepcopy(newArray)
        myScreen.txt_other_frequencies.setText(string_f)

        #Phase
        p_list=[]
        if myScreen.rd_app_other_pha.isChecked() :
            p_list = myScreen.txt_other_phases.text().split(' ')
        else:
            p_list = myScreen.txt_result_b_phase.text().split(' ')
        #p_list = myScreen.txt_result_b_phase.text().split(' ')
        list_freq_phase = p_list[:-1]
        newArray = []
        string_p = ''
        print('list_freq_phase', list_freq_phase)
        for x in list_freq_phase:
            #print('X', x, type(x))
            #if len(x) > 0 and x.isdigit():
            string_p += str(x) + ' '
            newArray.append(float(x))

        list_freq_phase = copy.deepcopy(newArray)
        myScreen.txt_other_phases.setText(string_p)


        #Electrodes
        e_list=[]
        if myScreen.rd_app_other_elec.isChecked():
            e_list = myScreen.txt_other_electrodes.text().split(' ')
        else:
            e_list = myScreen.txt_result_b_elec.text().split(' ')

        #e_list = myScreen.txt_result_b_elec.text().split(' ')
        list_good_sensors_pos = e_list[:-1]
        newArray = []
        string_e = ''
        for x in list_good_sensors_pos:
            #if len(x) > 0 and x.isdigit():
             string_e += str(x) + ' '
             newArray.append(float(x))
        list_good_sensors_pos = copy.deepcopy(newArray)
        myScreen.txt_other_electrodes.setText(string_e)

        print('Frec list :', list_freq_to_test)
        print('Asoc Phase:', list_freq_phase)
        print('Electrodes:', list_good_sensors_pos)

        if (len(list_freq_to_test)<=0 or len(list_freq_phase)<=0 ):
            print("There is no enough data to continue")
            working_any_thread = False
            refresh_window()
            return

        set_arduino()

        list_freq_detected = []
        list_s_power_freq = []
        if new_online_times <= myScreen.spin_times_app.value():
            online_times += 1

            current_freq_pos = 0
            while (current_freq_pos < len(list_freq_to_test)):
                #string = 'Freq: ' + str(list_freq_to_test[current_freq_pos]) + ', Phase: ' + str(
                #    list_freq_phase[current_freq_pos])
                # Configure Arduino
                string = str(list_freq_to_test[0]) + ' ' + str(list_freq_phase[0])
                string += ' ' + str(list_freq_to_test[1]) + ' ' + str(list_freq_phase[1])
                string += ' ' + str(list_freq_to_test[2]) + ' ' + str(list_freq_phase[2])
                string += ' ' + str(list_freq_to_test[3]) + ' ' + str(list_freq_phase[3])

                write_statusbar(string)
                # sleep(1)

                # Mode set on
                execute('sudo echo "179" > ' + str(ARDUINO_PORT) + '')


                #print(string)
                execute('sudo echo "' + string + ' " > ' + str(ARDUINO_PORT) + '')

                # gUSBamp trigger the led and record
                err, out = execute('sudo echo $$&')
                execute('sudo ./gUSBamp_files/gUSBampAPI_Demo ' + str(out), showresult=False)

                # Turn off leds
                execute('sudo echo "999" > /dev/ttyACM0')

                f, s_power = execute_CCA(time=online_times, concept='online', value=list_freq_to_test[current_freq_pos], good_sensor_list=list_good_sensors_pos, sensors_num=16)

                list_freq_detected.append(f)
                list_s_power_freq.append(round(s_power, 4))

                current_freq_pos += 1

            print('freq detected:', list_freq_detected)
            print('RCC by freq  :', list_s_power_freq)

            write_statusbar('Validation has finished')
        else:
            samplerate = 512
            num_sensors = 16
            num_totaltimes = myScreen.spin_times_app.value()
            freq_min = min(list_freq_to_test) #[0]
            freq_max = max(list_freq_to_test) #[-1]

            #print('num_totaltimes',num_totaltimes)
            #print('freq_min',freq_min)
            #print('freq_max',freq_max)

            app = AppManager()
            app.set_variables(sessionFolder, samplerate, num_sensors, num_totaltimes, freq_min, freq_max, list_freq_to_test)

            best_4_f_list, best_4_snr_list, score = app.find_best_4_freq(good_sensor_list=list_good_sensors_pos)

            string_f = ''
            for x in best_4_f_list[:4]:
                string_f += str(x) + ' '

            myScreen.txt_validation_result_b_elec.setText(string_f)
            score = round(float(score),2)

            myScreen.lbl_final_score.setText(str(score)+ lbl_percent)

            write_statusbar('Times for step 4 is up!')

        myScreen.lbl_validation_count_b_freq.setText('')
        myScreen.lbl_validation_count_b_freq.setText(str(online_times) + lbl_counter)

    except Exception as err:
        print(err)

    working_any_thread = False
    refresh_window()


def control_of_threads(step):
    global working_any_thread

    compose_session_cod()
    setWorkingFolder()

    refresh_window()

    if (not working_any_thread):
        write_statusbar('Thread launched')
        working_any_thread = True
        if step == 2:
            #pool.apply_async(Thread_step_1_best_elec, ('a_', 'b_'))
            Thread_step_2_best_elec('a_', 'b_')
        elif step == 3:
            # pool.apply_async(Thread_step_2_best_freq, ('a_', 'b_'))
            Thread_step_3_best_phase('a_', 'b_')
        elif step == 4:
            Thread_step_4_validation('a_', 'b_')



        #pool.close()
        #pool.join()
        refresh_window()


    else:
        write_statusbar('Program is working in Background')

    refresh_window()

def clear_best_frequencies():
    global freq_times,best_freq_list
    global elec_times, best_elect_list
    refresh_window()

    elec_times = 0
    best_elect_list = []
    myScreen.txt_result_b_elec.setText('')

    freq_times = 0
    best_freq_list = []
    myScreen.txt_result_b_freq.setText('')
    myScreen.lbl_count_b_freq.setText('0' + lbl_counter)
    refresh_window()

def clear_best_phases():
    global phase_times, best_phase_list
    refresh_window()

    phase_times = 0
    best_phase_list = []
    myScreen.txt_result_b_phase.setText('')
    myScreen.lbl_count_b_phase.setText('0' + lbl_counter)
    refresh_window()

def clear_online():
    global online_times, online_times_total
    refresh_window()

    online_times = 0
    myScreen.txt_validation_result_b_elec.setText('')
    myScreen.lbl_final_score.setText('0' + lbl_percent)
    myScreen.lbl_validation_count_b_freq.setText('0' + lbl_counter)
    refresh_window()


def set_click_connections():
    myScreen.btn_save_user.clicked.connect(setWorkingFolder)
    #myScreen.btn_launch_b_elec.clicked.connect(partial(control_of_threads,step=1))
    myScreen.btn_launch_b_freq.clicked.connect(partial(control_of_threads,step=2))
    myScreen.btn_launch_b_phase.clicked.connect(partial(control_of_threads,step=3))
    myScreen.btn_launch_online.clicked.connect(partial(control_of_threads,step=4))

    myScreen.btn_clear_freq.clicked.connect(clear_best_frequencies)
    myScreen.btn_clear_phase.clicked.connect(clear_best_phases)
    myScreen.btn_clear_online.clicked.connect(clear_online)
    myScreen.btn_exportJSON.clicked.connect(export_to_json)

    myScreen.rd_app_b_elec.toggled.connect(check_radiobuttons)
    myScreen.rd_app_b_freq.toggled.connect(check_radiobuttons)
    myScreen.rd_app_b_pha.toggled.connect(check_radiobuttons)

    myScreen.rd_app_other_elec.toggled.connect(check_radiobuttons)
    myScreen.rd_app_other_freq.toggled.connect(check_radiobuttons)
    myScreen.rd_app_other_pha.toggled.connect(check_radiobuttons)

def check_radiobuttons():
    if(myScreen.rd_app_other_elec.isChecked()):
        myScreen.txt_other_electrodes.setEnabled(True)
    else:
        myScreen.txt_other_electrodes.setEnabled(False)

    if(myScreen.rd_app_other_freq.isChecked()):
        myScreen.txt_other_frequencies.setEnabled(True)
    else:
        myScreen.txt_other_frequencies.setEnabled(False)

    if(myScreen.rd_app_other_pha.isChecked()):
        myScreen.txt_other_phases.setEnabled(True)
    else:
        myScreen.txt_other_phases.setEnabled(False)



#t_freq = threading.Thread(target=MyThread1, args=[])
#############

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv) #create a new instance of QApplication

    font = QFont("Helvetica")
    #font.setFamily(font.C defaultFamily())
    font.setPointSize(10.5)
    app.setFont(font)
    windows = QtWidgets.QMainWindow() #create an instance of QMainWindow

    myScreen.setupUi(windows) #create your widgets inside the window you just created.

    
    set_default_values_on_windows()
    set_click_connections()
    #launch_timer()
    windows.show() #show the window.
    windows.setWindowTitle(' SSVEPAdapt v1.0 by GNB-UAM | User adaptation for SSVEP-BCIs')
    sys.exit(app.exec_()) #Exit the program when you close the application window.

