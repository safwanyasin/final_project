# import RPi.GPIO as GPIO
# GPIO.setwarnings(False)

# import os, time

# edgetpu=0 # If Coral USB Accelerator connected, then make it '1' otherwise '0'

# m1_1 = 8
# m1_2 = 11
# m2_1 = 14 
# m2_2 = 15 
# cam_light = 17
# headlight_right = 18
# headlight_left = 27 
# sp_light=9 


# def init_gpio():
# 	GPIO.setmode(GPIO.BCM)
# 	GPIO.setup(m1_1,GPIO.OUT)
# 	GPIO.setup(m1_2,GPIO.OUT)
# 	GPIO.setup(m2_1,GPIO.OUT)
# 	GPIO.setup(m2_2,GPIO.OUT)
# 	GPIO.setup(cam_light,GPIO.OUT)
# 	GPIO.setup(headlight_right,GPIO.OUT)
# 	GPIO.setup(headlight_left,GPIO.OUT)
# 	GPIO.setup(sp_light,GPIO.OUT)
	

# def back():
#     print("moving back!!!!!!")
#     GPIO.output(m1_1, False)
#     GPIO.output(m1_2, True)
#     GPIO.output(m2_1, True)
#     GPIO.output(m2_2, False)
    
# def right():
# 	GPIO.output(m1_1, True)
# 	GPIO.output(m1_2, False)
# 	GPIO.output(m2_1, True)
# 	GPIO.output(m2_2, False)

# def left():
# 	GPIO.output(m1_1, False)
# 	GPIO.output(m1_2, True)
# 	GPIO.output(m2_1, False)
# 	GPIO.output(m2_2, True)
	
# def forward():
# 	GPIO.output(m1_1, True)
# 	GPIO.output(m1_2, False)
# 	GPIO.output(m2_1, False)
# 	GPIO.output(m2_2, True)
	
# def stop():
# 	GPIO.output(m1_1, False)
# 	GPIO.output(m1_2, False)
# 	GPIO.output(m2_1, False)
# 	GPIO.output(m2_2, False)

# def speak_tts(text,gender):
# 	cmd="python /var/www/html/earthrover/speaker/speaker_tts.py '" + text + "' " + gender + " &"
# 	os.system(cmd)
	
# def camera_light(state):
# 	if(state=="ON"):
# 		GPIO.output(cam_light, True)
# 		#print("light on")
# 	else:
# 		GPIO.output(cam_light, False)
# 		#print("light off")
		
# def head_lights(state):
# 	if(state=="ON"):
# 		GPIO.output(headlight_left, True)
# 		GPIO.output(headlight_right, True)
# 		#print("light on")
# 	else:
# 		GPIO.output(headlight_left, False)
# 		GPIO.output(headlight_right, False)
# 		#print("light off")
		
# def red_light(state):
# 	if(state=="ON"):
# 		GPIO.output(sp_light, True)
# 		#print("light on")
# 	else:
# 		GPIO.output(sp_light, False)
# 		#print("light off")
	


import os, time

edgetpu = 0  # If Coral USB Accelerator connected, then make it '1' otherwise '0'

m1_1 = 8
m1_2 = 11
m2_1 = 14 
m2_2 = 15 
cam_light = 17
headlight_right = 18
headlight_left = 27 
sp_light = 9 

def init_gpio():
    print("init_gpio")

def back():
    print("moving back!!!!!!")
    print("Setting m1_1 to LOW")
    print("Setting m1_2 to HIGH")
    print("Setting m2_1 to HIGH")
    print("Setting m2_2 to LOW")

def right():
    print("moving right!!!!!!")
    print("Setting m1_1 to HIGH")
    print("Setting m1_2 to LOW")
    print("Setting m2_1 to HIGH")
    print("Setting m2_2 to LOW")

def left():
    print("moving left!!!!!!")
    print("Setting m1_1 to LOW")
    print("Setting m1_2 to HIGH")
    print("Setting m2_1 to LOW")
    print("Setting m2_2 to HIGH")

def forward():
    print("moving forward!!!!!!")
    print("Setting m1_1 to HIGH")
    print("Setting m1_2 to LOW")
    print("Setting m2_1 to LOW")
    print("Setting m2_2 to HIGH")

def stop():
    print("stopping the robot!!!!!!")
    print("Setting m1_1 to LOW")
    print("Setting m1_2 to LOW")
    print("Setting m2_1 to LOW")
    print("Setting m2_2 to LOW")

def speak_tts(text, gender):
    cmd = "python /var/www/html/earthrover/speaker/speaker_tts.py '" + text + "' " + gender + " &"
    os.system(cmd)

def camera_light(state):
    if state == "ON":
        print("Turning camera light ON")
    else:
        print("Turning camera light OFF")

def head_lights(state):
    if state == "ON":
        print("Turning headlights ON")
    else:
        print("Turning headlights OFF")

def red_light(state):
    if state == "ON":
        print("Turning red light ON")
    else:
        print("Turning red light OFF")
