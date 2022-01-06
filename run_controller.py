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

class CityFlowEnv:
    #Todo: implement
    pass


def build_env(config,cityflow_config,model_dir,result_dir,
                num_step,episodes,agent,phase_list,interval,phase_step,save_freq,
                update_target_model_freq,learning_start,update_model_freq,algo):
    # build cityflow environment
    cityflow_config["saveReplay"] = False
    json.dump(cityflow_config, open(config["cityflow_config_file"], 'w'))
    # create env
    env = CityFlowEnv(config)
    # crete new folder if not exist
    if not os.path.exists("model"):
        os.makedirs("model")
    if not os.path.exists("result"):
        os.makedirs("result")
    # create model | result folder
    os.makedirs(model_dir)
    os.makedirs(result_dir)

    # training
    total_step = 0
    episode_rewards = []
    episode_scores = []
    # show progress in show way
    with tqdm(total=episodes * num_step) as progress_bar:
        for i in range(episodes):
            # reset before each episodes
            env.reset()
            state = env.get_state()
            episode_length = 0
            episode_reward = 0
            episode_score = 0
            # run untill rech max number of steps
            while episode_length < num_step:
                # get action from agent
                action = agent.choose_action(state)
                # get the phase index of action
                action_phase = phase_list[action]
                # min action time (in number of states)
                for _ in range(interval):
                    next_state, reward = env.step(action_phase)  # one step
                # update 
                episode_length += 1
                total_step += 1
                episode_reward += reward
                episode_score += env.get_score()

                for _ in range(phase_step - 1):
                    next_state, reward_ = env.step(action_phase)
                    reward += reward_

                reward /= phase_step

                progress_bar.update(1)
                # store to replay buffer
                agent.remember(state, action_phase, reward, next_state)

                state = next_state

                # training
                if total_step > learning_start and total_step % update_model_freq == 0:
                    agent.replay()

                # update target Q netwark
                if total_step > learning_start and total_step % update_target_model_freq == 0:
                    agent.update_target_network()

                progress_bar.set_description(
                    "total_step:{}, episode:{}, episode_step:{}, reward:{}".format(total_step, i + 1,
                                                                                    episode_length, reward))


            # save episode rewards
            episode_rewards.append(episode_reward)
            episode_scores.append(episode_score)
            print("score: {}, mean reward:{}".format(episode_score, episode_reward / num_step))

            # save model
            if (i + 1) % save_freq == 0:
                #Todo: save model

                # save reward to file
                df = pd.DataFrame({"rewards": episode_rewards})
                df.to_csv(result_dir + '/rewards.csv', index=None)

                df = pd.DataFrame({"rewards": episode_scores})
                df.to_csv(result_dir + '/scores.csv', index=None)

        # save figure
        plot_data_lists([episode_rewards], ['episode reward'], figure_name=result_dir + '/rewards.pdf')
        plot_data_lists([episode_scores], ['episode score'], figure_name=result_dir + '/scores.pdf')







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
    phase_step = args.phase_step
    save_freq = args.save_freq
    


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