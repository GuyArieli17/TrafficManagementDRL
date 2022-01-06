import argparse
import json
import logging
import os
import numpy as np
from datetime import datetime
from utils import parse_roadnet, plot_data_lists
from tqdm import tqdm

import pandas as pd
import time

def run():
    logging.getLogger().setLevel(logging.INFO)
    date = datetime.now().strftime('%Y%m%d_%H%M%S')
    parser = argparse.ArgumentParser()
    # config file path 
    parser.add_argument('--config', type=str, default='config/global_config.json', help='config file')
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

    print(args)
    # # get config file path
    # config = json.load(open(args.config))
    # # add number of steps
    # config["num_step"] = args.num_step
    # # load config file
    # cityflow_config = json.load(open(config['cityflow_config_file']))
    # # get rodnet file path
    # roadnetFile = cityflow_config['dir'] + cityflow_config['roadnetFile']
    # #
    # config["lane_phase_info"] = parse_roadnet(roadnetFile)


if __name__ == '__main__':
    run()