import cityflow
import pandas as pd
import os
import json
import math
import numpy as np

class CityFlowEnv:

    def __init__(self, config):
        # config env as configed
        self.eng = cityflow.Engine(config['cityflow_config_file'], thread_num=config['thread_num'])
        # save information
        self.config = config
        self.num_step = config['num_step']
        self.state_size = len(config['lane_phase_info'][config["intersection_id"]]['start_lane']) + 1
        self.lane_phase_info = config['lane_phase_info']
        self.intersection_id = config["intersection_id"]
        self.start_lane = self.lane_phase_info[self.intersection_id]['start_lane']
        self.phase_list = self.lane_phase_info[self.intersection_id]["phase"]
        self.phase_startLane_mapping = self.lane_phase_info[self.intersection_id]["phase_startLane_mapping"]

        self.current_phase = self.phase_list[0]
        self.current_phase_time = 0
        self.yellow_time = 5
        self.state_store_i = 0

        self.phase_log = []

    def reset(self):
        self.eng.reset()

    def step(self, next_phase):
        if self.current_phase == next_phase:
            self.current_phase_time += 1
        else:
            self.current_phase = next_phase
            self.current_phase_time = 1

        self.eng.set_tl_phase(self.intersection_id, self.current_phase)  # set phase of traffic light
        self.eng.next_step()
        self.phase_log.append(self.current_phase)

        return self.get_state(), self.get_reward()  # return next_state and reward

    def get_state(self):
        state = {}
        state['lane_vehicle_count'] = self.eng.get_lane_vehicle_count()  # {lane_id: lane_count, ...}
        state['start_lane_vehicle_count'] = {lane: self.eng.get_lane_vehicle_count()[lane] for lane in self.start_lane}
        state['lane_waiting_vehicle_count'] = self.eng.get_lane_waiting_vehicle_count()  # {lane_id: lane_waiting_count, ...}
        state['lane_vehicles'] = self.eng.get_lane_vehicles()  # {lane_id: [vehicle1_id, vehicle2_id, ...], ...}
        state['vehicle_speed'] = self.eng.get_vehicle_speed()  # {vehicle_id: vehicle_speed, ...}
        state['vehicle_distance'] = self.eng.get_vehicle_distance()  # {vehicle_id: distance, ...}
        state['current_time'] = self.eng.get_current_time()
        state['current_phase'] = self.current_phase
        state['current_phase_time'] = self.current_phase_time

        state_pre = self.waiting_count_pre_1()
        return_state = np.array(list(state_pre) + [state['current_phase']])
        return_state = np.reshape(return_state, [1, self.state_size])

        return return_state

    def waiting_count_pre_1(self):
        state_pre = list(self.eng.get_lane_waiting_vehicle_count().values())
        state=np.zeros(8)
        state[0]=state_pre[1]+state_pre[15]
        state[1]=state_pre[3]+state_pre[13]
        state[2]=state_pre[0]+state_pre[14]
        state[3]=state_pre[2]+state_pre[12]
        state[4]=state_pre[1]+state_pre[0]
        state[5]=state_pre[14]+state_pre[15]
        state[6]=state_pre[3]+state_pre[2]
        state[7]=state_pre[12]+state_pre[13]
        return state


        # def get_reward(self):

    def waiting_count_pre_2(self):
        state_pre=state_pre = list(self.eng.get_lane_waiting_vehicle_count().values())


    #     # a sample reward function which calculates the total of waiting vehicles
    #     lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
    #     lane_waiting_vehicle_count_list = list(lane_waiting_vehicle_count.values())
    #     reward = -1 * ( sum(lane_waiting_vehicle_count_list)/len(lane_waiting_vehicle_count_list) + max(lane_waiting_vehicle_count_list) )
    #     return reward

    def get_reward(self):
        mystate = self.get_state()
        # reward function
        lane_vehicle_count = mystate[0][0:8]
        vehicle_velocity = self.eng.get_vehicle_speed()
        # reward = sum(list(vehicle_velocity.values()))
        reward = -max(lane_vehicle_count)
        # reward_sig = 2 / ((1 + math.exp(-1 * reward)))
        return reward

    def get_score(self):
        lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        reward = -sum(list(lane_waiting_vehicle_count.values()))
        metric = 1 / ((1 + math.exp(-1 * reward)) * self.config["num_step"])
        return metric

    def log(self):
        if not os.path.exists(self.config['replay_data_path']):
            os.makedirs(self.config["replay_data_path"])
        df = pd.DataFrame({self.intersection_id: self.phase_log[:self.num_step]})
        df.to_csv(os.path.join(self.config['replay_data_path'], 'signal_plan.txt'), index=None)