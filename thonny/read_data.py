# micropython code for reading accelerometer data
from imu import MPU6050
import time
from machine import Pin, UART, I2C

i2c = I2C(0, sda=Pin(0), scl=Pin(1), freq=400000)
imu = MPU6050(i2c)

def set_baseline(i):
    baseline_x = []
    baseline_y = []
    baseline_z = []

    while (i != 0):
        (x, y, z) = imu.accel.xyz
        baseline_x.append(x)
        baseline_y.append(y)
        baseline_z.append(z)
        i -= 1
        
    base_x = sum(baseline_x) / len(baseline_x)
    base_y = sum(baseline_y) / len(baseline_y)
    base_z = sum(baseline_z) / len(baseline_z)
    
    return base_x, base_y, base_z

def record_data():
    base_x, base_y, base_z = set_baseline(200)

    threshold = .02
    end_threshold = .02
    record = False
    cnt = 0
    prev = None
    record_cnt = 0
        
    x_val = []
    y_val = []
    z_val = []

    while True:
        x, y, z = imu.accel.xyz
        
        if record:
            x_val.append(x)
            y_val.append(y)
            z_val.append(z)
            print(imu.accel.xyz, end='\n')
        elif record_cnt > 2 and len(x_val) > 10:
            print("data point recorded")
            cnt += 1
            x_val = []
            y_val = []
            z_val = []
                       
        if (not base_x - threshold < x < base_x + threshold) or (not base_y - threshold < y < base_y + threshold) or (not base_z - threshold < z < base_z + threshold):
            record = True
            record_cnt = 0
        
        if (base_x - end_threshold < x < base_x + end_threshold) or (base_y - end_threshold < y < base_y + end_threshold) or (base_z - end_threshold < z < base_z + end_threshold):
            record = False
            record_cnt += 1
            
        time.sleep(0.2)

# remember to set baseline position starting gesture
data = record_data()