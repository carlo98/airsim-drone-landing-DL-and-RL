import time
import os
import csv
import airsim
from tensorflow.keras.models import load_model as tf_load_model
import numpy as np

MIN_ALTITUDE = 10
MAX_ALTITUDE = 15
NUM_EPISODES = 100

#load pre-trained model
def load_trained_model():
    loaded_model = tf_load_model("./models/modelVerticalNN.h5")
    print("Model restored.")
    return loaded_model

def save_on_file(state, action):
    try:
        dataF = open("data/landingData.csv","a")
        spamwriter = csv.writer(dataF, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        string = [state, action]
        spamwriter.writerow(string)
        dataF.close()
    except IOError:
        print("File not found.")

def go_back(client, home):
    client.moveToPositionAsync(home.x_val, home.y_val, -1*np.random.randint(MIN_ALTITUDE,MAX_ALTITUDE+1), 5).join()
    time.sleep(0.5)
    client.moveByVelocityAsync(0, 0, -0.0, 5).join()

def interpret_action(action):
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


def get_action_data(curr_state):
    if curr_state <= 0.1:
        action_index = 3
    elif curr_state >= 4:
        action_index = 2
    elif curr_state >= 1.8:
        action_index = 0
    elif curr_state < 1.8 and curr_state >= 0.7: 
        action_index = 1
    else:
        action_index = 4

    return action_index

def get_action_test(curr_state):
    actions_index = my_model.predict(np.array([[curr_state]]))[0][0]
    print("Actions: ",actions_index)

    return round(actions_index)
    

# Load model if you are testing
my_model = load_trained_model()

# init drone
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

#taking home position
home = client.getMultirotorState().kinematics_estimated.position

cnt = 1

while cnt <= NUM_EPISODES:
    print("### NEW EPISODE: ",cnt)
    go_back(client, home)

    curr_state = 10.0

    while curr_state > 0.1:
        curr_state = np.abs(client.getLidarData().pose.position.z_val)

        ## Test
        action_index = get_action_test(curr_state)
      
        #take data
        #action_index = get_action_data(curr_state)

        next_action = interpret_action(action_index)

        new_vel_x = next_action[0]
        new_vel_y = next_action[1]
        new_vel_z = next_action[2]
        trigger = next_action[3]
        print("  ====== moving at (" + str(new_vel_x) + " " + str(new_vel_y) + " " + str(new_vel_z)  + "), trigger ",trigger)
        client.moveByVelocityAsync(new_vel_x, new_vel_y, new_vel_z, 1).join()

        # Test -- Not working very well
        new_state = np.abs(client.getLidarData().pose.position.z_val)
        if new_state <= 0.1:
            if trigger:
                print("Landed.")
                break
            elif new_vel_z <= 0.3:
                print("Moving near ground.")
            else:
                print("Collision.")
                break   
        elif new_state > 0.1 and trigger:
            print("Error, not landed.")
            break

        time.sleep(0.01)

        #save_on_file(curr_state, action_index)
        
    cnt += 1
