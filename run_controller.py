import argparse
import json
import logging
import os
from typing import Match
import numpy as np
from datetime import datetime
from utils import parse_roadnet, plot_data_lists
from tqdm import tqdm

import pandas as pd
import time

def build_env():
    # Todo: Implement
    pass

def interface():
    # Todo: Implement
    pass


def run():
    logging.getLogger().setLevel(logging.INFO)
    date = datetime.now().strftime('%Y%m%d_%H%M%S')
    parser = argparse.ArgumentParser()
    # config file path 
    parser.add_argument('--config', type=str, default='config/global.json', help='config file')
    # model we train
    parser.add_argument('--algo', type=str, default='DQN', choices=['DQN', 'DDQN', 'DuelDQN'],
                        help='choose an algorithm')
    # shold show interface
    parser.add_argument('--inference', action="store_true", help='inference or training')
    # number of epochs
    parser.add_argument('--epoch', type=int, default=1000, help='number of training epochs')
    # number of epoch in each epoch
    parser.add_argument('--num_step', type=int, default=200,
                        help='number of timesteps for one episode, and for inference')
    # how frequently save model
    parser.add_argument('--save_freq', type=int, default=1, help='model saving frequency')
    # change batch size
    parser.add_argument('--batch_size', type=int, default=30, help='batchsize for training')
    # time for each phase
    parser.add_argument('--phase_step', type=int, default=5, help='time of one phase')

    #namespace of all parser information
    args = parser.parse_args()

    # ------- Get All information from user ------- 
    EPISODES = args.epoch
    learning_start = 200
    update_model_freq = args.batch_size
    update_target_model_freq = 200
    interval = 1
    config_path = args.config
    num_step = args.num_step
    algo_str = args.algo


    # get config file path
    config = json.load(open(config_path))
    # add number of steps
    config["num_step"] = num_step
    # load config file
    cityflow_config = json.load(open(config['cityflow_config_file']))
    
    # get rodnet file path
    roadnetFile = cityflow_config['dir'] + cityflow_config['roadnetFile']


    # dict with information for each intersection
    lane_info = parse_roadnet(roadnetFile)
    # set main intesection_id
    intersection_id = "intersection_1_1"
    # how mant in road exist's + none action
    action_space = len(lane_info[intersection_id]['start_lane']) +1
    # get phase from main intersection
    phase_list = lane_info[intersection_id]['phase']
    # get model and result path
    model_dir = f"model/{algo_str}_{date}"
    result_dir = f"result/{algo_str}_{date}"

    # Todo: create agent for each input
    if algo_str == 'DQN':
        agent = None

    # ----- set config dict (json) -----
    config["action_size"] = len(phase_list)
    config["batch_size"] = args.batch_size
    config["intersection_id"] = intersection_id
    config["lane_phase_info"] = lane_info
    config["state_size"] = action_space
    config["result_dir"] = result_dir

    if not args.inference:
        build_env()
    else:
        interface()


if __name__ == '__main__':
    run()