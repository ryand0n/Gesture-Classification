import serial
import time
import serial.tools.list_ports
import matplotlib.pyplot as plt
import numpy as np
import q as cute
from itertools import count

# can be used to automate finding right COM port
def get_ports():
    ports = serial.tools.list_ports.comports()
    return ports

print([str(port) for port in get_ports()])