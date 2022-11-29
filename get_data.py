import serial
import time
import serial.tools.list_ports
import matplotlib.pyplot as plt
import numpy as np
import q as cute
from itertools import count
import csv
from model.build_model import predict

# can be used to automate finding right COM port
def get_ports():
    ports = serial.tools.list_ports.comports()
    return ports

def find_pico(ports_found):
    comm_port = 'None'
    num_connections = len(ports_found)

    for i in range(0,num_connections):
        port = ports_found[i]
        str_port = str(port)
        # change this to whatever device name
        if 'USB Serial Device' in str_port:
            comm_port = str_port.split('-')    
    return comm_port[0].strip()

# make a serial object
arduino = serial.Serial(port=find_pico(get_ports()), baudrate=115200, timeout=.1)


index = count()


def get_data():
    
    input = arduino.readline()

    if len(input) > 0:
        return input.decode('utf-8').strip('\r\n()').split(', ')
    else:
        return None

    data = []
    for _ in range(10):
        input = arduino.readlines(10000)
        data += [[float(val) for val in d.decode('utf-8').strip('\n').strip('\r').strip('(').strip(')').split(', ')] for d in input]

    data = np.array(data)
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]

    idxs = [next(index) * 0.01 for _ in range(len(x))]
    print(x,y,z)

def plot():
    plt.style.use('fivethirtyeight')
    fig = plt.figure()
    ax = fig.add_subplot(111)

    i = 0
    idxs = cute.Queue()
    x_vals = cute.Queue()
    y_vals = cute.Queue()
    z_vals = cute.Queue()

    idxs.add(i)
    x_vals.add(float(x))
    y_vals.add(float(y))
    z_vals.add(float(z))

    ax.plot(idxs.queue, x_vals.queue, color='b', label='x')
    ax.plot(idxs.queue, y_vals.queue, color='r', label='y')
    ax.plot(idxs.queue, z_vals.queue, color='g', label='z')
    ax.legend()

    fig.canvas.draw()

    ax.set_xlim(left=max(0, i - 50), right=i + 50)
    fig.show()
    plt.pause(0.05)
    i += 1

    if idxs.size >= 50:
        idxs.remove()
    if x_vals.size >= 50:
        x_vals.remove()
    if y_vals.size >= 50:
        y_vals.remove()
    if z_vals.size >= 50:
        z_vals.remove()

def collect_data(i, path, predicter=False, model=None):
    """
    i: how many data points you want to collect
    path: where you want to write your csv file to
    predicter: whether or not you want to make predictions on incoming data
    """
    record = False
    index = 0
    all_data = []
    x_data = []
    y_data = []
    z_data = []


    while index < i:
        x,y,z = None,None,None
        data = get_data()

        if data is not None:
            print(data)

        if data is not None and data[0] == 'data point recorded':
            record = False

        if data is not None and data[0] == '\x04Traceback (most recent call last):':
            break

        if data is not None and record and data[0] != 'data point recorded':
            x,y,z = data
            x,y,z = float(x), float(y), float(z)
            x_data.append(x)
            y_data.append(y)
            z_data.append(z)
        elif not record:
            all_data.append({'index':index, 'x':x_data, 'y':y_data, 'z':z_data})

            if predicter:
                print(predict(model, {'index':index, 'x':x_data, 'y':y_data, 'z':z_data}))

            index += 1
            x_data = []
            y_data = []
            z_data = []
            record = True
            print("data point recorded")
            print(f"index: {index}")
        else:
            continue

    csv_columns = ['index', 'x', 'y', 'z']
    dict_data = all_data
    csv_file = path

    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")

    return dict_data
