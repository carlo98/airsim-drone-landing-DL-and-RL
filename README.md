# airsim-drone-landing-DL-and-RL
Airsim multirotor landing in two phases using deep reinforcement learning and deep learning.

droneRL_landingVerticalLidar.py: is based on the RL code found here: https://adventuresinmachinelearning.com/reinforcement-learning-tensorflow/

droneNN_landing.py: complete landing using deep learning, trained models can be found in "models". If you want to train you can find two notebooks (one for horizontal and one for vertical); for the horizontal phase you'll need to take images and split them in five different folder (up, down, left, right, trigger), for the vertical phase you can find a method, that takes data and saves it as csv file, in droneNN_landingVertical.py.

Airsim repo: https://github.com/microsoft/AirSim

