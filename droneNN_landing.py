import airsim
import cv2
import time
import os
import numpy as np
from tensorflow.keras.models import load_model as tf_load_model

MIN_ALTITUDE = 5
MAX_ALTITUDE = 8

def go_back(client, home):
    client.moveToPositionAsync(home.x_val, home.y_val, -1*np.random.randint(MIN_ALTITUDE,MAX_ALTITUDE+1), 5).join()
    time.sleep(0.5)
    client.moveToPositionAsync(np.random.randint(home.x_val-3,home.x_val+4), np.random.randint(home.y_val-3,home.y_val+4), -1*np.random.randint(MIN_ALTITUDE,MAX_ALTITUDE+1), 5).join()
    time.sleep(0.5)
    client.moveByVelocityAsync(0, 0, -0.0, 5).join()

def is_in_bounds_2d(pos, home):
    return distance_2d(pos, home) < MAX_RADIUS

def distance_2d(pos, home):
    return np.sqrt((home.x_val-pos.x_val)**2 + (home.y_val-pos.y_val)**2)

#load horizontal pre-trained model
def load_hor_trained_model():
    loaded_model = tf_load_model("./models/modelHorizontalImage.h5")
    print("Model restored.")
    return loaded_model

#load vertical pre-trained model
def load_ver_trained_model():
    loaded_model = tf_load_model("./models/modelVerticalNN_2.h5")
    print("Model restored.")
    return loaded_model

def get_image(client):
    image_buf = np.zeros((1, 432 , 768, 4))
    image_response = client.simGetImages([airsim.ImageRequest(3, airsim.ImageType.Scene, False, False)])[0]
    png = client.simGetImages([airsim.ImageRequest(3, airsim.ImageType.Scene)])[0]
    image1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
    image_rgba = image1d.reshape(image_response.height, image_response.width, 3)
    image_rgba = cv2.cvtColor(image_rgba,cv2.COLOR_RGBA2BGR)
    image_buf = image_rgba.copy()
    image_buf = cv2.resize(image_buf,(150,150))

    return image_buf, png

def get_action_test(curr_state, my_model_ver):
    actions_index = my_model_ver.predict(np.array([[curr_state]]))[0][0]
    print("Actions: ",actions_index)

    return round(actions_index)

def interpret_action_hor(action):
    if action == 0:
        quad_offset = (-1, 0, 0, 0)
    elif action == 1:
        quad_offset = (0, -1, 0, 0)
    elif action == 2:
        quad_offset = (0, 1, 0, 0)
    elif action == 3:
        quad_offset = (0, 0, 0, 1)
    elif action == 4:
        quad_offset = (1, 0, 0, 0)
    return quad_offset

def interpret_action_ver(action):
    if action == 0:
        quad_offset = (0, 0, 1, 0)
    elif action == 1:
        quad_offset = (0, 0, 0.3, 0)
    elif action == 2:
        quad_offset = (0, 0, 2, 0)
    elif action == 3:
        quad_offset = (0, 0, 0, 1)
    elif action == 4:
        quad_offset = (0, 0, 0.1, 0)
    return quad_offset

def testNetworks():
    flag_stop = "a"
    snapshot_index = 0

    my_model_hor = load_hor_trained_model()
    my_model_ver = load_ver_trained_model()
    
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)

    home = client.getMultirotorState().kinematics_estimated.position

    while flag_stop != "stop":
        go_back(client, home)
        trigger_hor = 0
        trigger_ver = 0
        step = 0
    
        while True and step < 40:

            #Stop and study data
            client.moveByVelocityAsync(0, 0, -0.0, 5)

            #Horizontal movement
            if not trigger_hor:
                current_position = client.getMultirotorState().kinematics_estimated.position
                observe, png = get_image(client)
                observe = np.array(observe, dtype=np.float32)
                observe = observe.reshape(1,150,150,3)
                actions_index = my_model_hor.predict(observe)[0]

                action_index = np.argmax(actions_index)
                next_action = interpret_action_hor(action_index)
            #Vertical movement
            else:
                curr_state = np.abs(client.getLidarData().pose.position.z_val)
                action_index = get_action_test(curr_state, my_model_ver)
                next_action = interpret_action_ver(action_index)
                
            new_vel_x = next_action[0]
            new_vel_y = next_action[1]
            new_vel_z = next_action[2]
            trigger = next_action[3]

            if trigger and not trigger_hor:
                trigger_hor = 1
            elif trigger and not trigger_ver:
                trigger_ver = 1

            print(" Action index: ",action_index," ====== moving at (" + str(new_vel_x) + " " + str(new_vel_y) + " " + str(new_vel_z)  + ")")
            client.moveByVelocityAsync(new_vel_x, new_vel_y, new_vel_z, 1).join()

            time.sleep(0.001)

            step += 1

            #Vertical reset
            if trigger_hor:
                new_state = np.abs(client.getLidarData().pose.position.z_val)
                if new_state <= 0.1:
                    if trigger_ver:
                        print("Landed.")
                        break
                    elif new_vel_z <= 0.3:
                        print("Moving near ground.")
                    else:
                        print("Collision.")
                        break   
                elif new_state > 0.1 and trigger_ver:
                    print("Error, not landed.")
                    break
        
        flag_stop = input("Digitare stop se si vuole terminare.")
   

        

if __name__ == "__main__":
    testNetworks()

