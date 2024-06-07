import serial
import random
import time
ser = serial.Serial('COM6', 9600) 
def send_pwm_values():
    # Open serial connection to Arduino Nano
    # Send each PWM value to Arduino
    # for pwm in pwm_values:
    #     print("sending", pwm)
    #     ser.write(bytes(str(pwm), 'utf-8'))
    #     time.sleep(1)  # Delay to allow Arduino to process
    #     data = ser.readline()
    #     return data
    # pwm_values = [random.randint(0, 255), random.randint(0, 255), random.randint(0,255)] 
    pwm_values = [0,0,0]
    print("sending", pwm_values)
    ser.write(bytes(str(pwm_values) + '\n', 'utf-8'))
    # time.sleep(1)  # Delay to allow Arduino to process
    data = ser.readline()
    return data
    # Close serial connection
    # ser.close()

# Example usage
 # Example PWM values

 # Change 'COM3' to your Arduino's port
while True:
    value = send_pwm_values()
    print(value)
    time.sleep(1)
# import serial
# import serial.tools.list_ports
# import time

# def list_available_ports():
#     ports = list(serial.tools.list_ports.comports())
#     available_ports = [p.device for p in ports]
#     return available_ports

# def connect_serial(port, baudrate, retries=3, delay=2):
#     for i in range(retries):
#         try:
#             ser = serial.Serial(port, baudrate)
#             print(f"Successfully connected to {port}")
#             return ser
#         except serial.SerialException as e:
#             print(f"Attempt {i+1} failed: {e}")
#             time.sleep(delay)
#     raise serial.SerialException(f"Failed to connect to {port} after {retries} attempts")

# available_ports = list_available_ports()
# print("Available ports:", available_ports)

# if 'COM7' in available_ports:
#     try:
#         ser = connect_serial('COM7', 9600)
#     except serial.SerialException as e:
#         print(f"Error opening port: {e}")
# else:
#     print("COM6 is not available. Please check the connection and try again.")
