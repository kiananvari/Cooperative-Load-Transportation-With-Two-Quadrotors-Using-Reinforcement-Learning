# Add Webots controlling libraries
from controller import Robot
from controller import Supervisor
import struct 


# Some general libraries
import os
import time
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import math

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# Stable_baselines3
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

# Create an instance of robot
robot = Robot()

# Seed Everything
seed = 42
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.loss_values = []  # Add this line to store loss values


    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              
              mean_reward = np.mean(y[-100:])
            #   print("mean_reward:-------------------------------> ", mean_reward)
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print("----------------------------------------------------------------------------------------------")
                    print(f"Saving new best model to {self.save_path}")
                    print("----------------------------------------------------------------------------------------------")
                  self.model.save(self.save_path)

        return True


class Environment(gym.Env, Supervisor):
    """The robot's environment in Webots."""
    
    def __init__(self):
        super().__init__()
                
        # General environment parameters

        # self.max_speed = 1.5 # Maximum Angular speed in rad/s
        # self.start_coordinate = np.array([-2.60, -2.96])
        # self.destination_coordinate = np.array([-0.03, 2.72]) # Target (Goal) position
        # self.reach_threshold = 0.06 # Distance threshold for considering the destination reached.
        # obstacle_threshold = 0.1 # Threshold for considering proximity to obstacles.
        # self.obstacle_threshold = 1 - obstacle_threshold

        # self.sampling_period = 200 # in ms
        self.sampling_period = int(robot.getBasicTimeStep()) 
        self.startinig_coordinates = np.array([0, 0])   

        # self.destination_coordinates = np.array([30, -30]) # Target (Goal) position
        self.destination_coordinates = np.array([[30, -30], [-30, 30], [30, 30], [-30, -30]])
        self.floor_size = np.linalg.norm([100, 100])

        # Activate Devices
        #~~ 1) Wheel Sensors
        self.front_left_motor = robot.getDevice('front left propeller')
        self.front_right_motor = robot.getDevice('front right propeller')
        self.rear_left_motor = robot.getDevice('rear left propeller')
        self.rear_right_motor = robot.getDevice('rear right propeller')

        # Set the motors to rotate for ever
        self.front_left_motor.setPosition(float('inf'))
        self.front_right_motor.setPosition(float('inf'))
        self.rear_left_motor.setPosition(float('inf'))
        self.rear_right_motor.setPosition(float('inf'))

        # starting velocity
        self.front_left_motor.setVelocity(1.0)
        self.front_right_motor.setVelocity(1.0)
        self.rear_left_motor.setVelocity(1.0)
        self.rear_right_motor.setVelocity(1.0)


        #~~ 2) LED
        self.front_left_led = robot.getDevice('front left led')
        self.front_right_led = robot.getDevice('front right led')

        #~~ 3) GPS Sensor
        self.gps = robot.getDevice("gps")
        self.gps.enable(self.sampling_period)

        #~~ 4) Compass Sensor
        self.compass = robot.getDevice("compass")
        self.compass.enable(self.sampling_period)

        #~~ 5) Gyro Sensor
        self.gyro = robot.getDevice("gyro")
        self.gyro.enable(self.sampling_period)

        #~~ 6) IMU Sensor
        self.imu = robot.getDevice("inertial unit")
        self.imu.enable(self.sampling_period)

        #~~ 7) Camera Sensor
        self.camera = robot.getDevice("camera")
        self.camera.enable(self.sampling_period)

        #~~ 8) Camera Motors
        self.camera_roll_motor = robot.getDevice("camera roll")
        self.camera_pitch_motor = robot.getDevice("camera pitch")
        self.camera_yaw_motor = robot.getDevice("camera yaw")

        # #~~ 9) Enable Touch Sensor
        self.touch11 = robot.getDevice("TS11")
        self.touch11.enable(self.sampling_period)
        self.touch12 = robot.getDevice("TS12")
        self.touch12.enable(self.sampling_period)
        # self.touch21 = robot.getDevice("TS21")
        # self.touch21.enable(self.sampling_period)
        # self.touch22 = robot.getDevice("TS22")
        # self.touch22.enable(self.sampling_period)

        # #~~ 10) Enable Receiver/Emitter

        self.receiver = robot.getDevice("receiver") # Retrieve the receiver and emitter by device name
        self.emitter = robot.getDevice("emitter")
        self.receiver.enable(self.sampling_period)



        # Constants, empirically found.
        self.k_vertical_offset = 0.6;   # Vertical offset where the robot actually targets to stabilize itself.
        self.k_vertical_p = 3.0;        # P constant of the vertical PID.
        self.k_vertical_thrust = 68.5;  # with this thrust, the drone lifts.
        self.k_roll_p = 50.0;           # P constant of the roll PID.
        self.k_pitch_p = 30.0;          # P constant of the pitch PID.

        self.target_altitude = 2.0;     # The target altitude. Can be changed by the user.

        #Space
        self.action_space = spaces.Discrete(6)
        # self.observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(14,), dtype=np.float32)

        self.max_steps = 1000

        # Reset
        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        
        
    def normalizer(self, value, min_value, max_value):
        """
        Performs min-max normalization on the given value.

        Returns:
        - float: Normalized value.
        """
        normalized_value = (value - min_value) / (max_value - min_value)        
        return normalized_value
        

    # def get_distance_to_goal(self):
    #     """
    #     Calculates and returns the normalized distance from the robot's current position to the goal.
        
    #     Returns:
    #     - numpy.ndarray: Normalized distance vector.
    #     """
        
    #     gps_value = self.gps.getValues()[0:2]
    #     current_coordinate = np.array(gps_value)
    #     distance_to_goal = np.linalg.norm(self.destination_coordinate - current_coordinate)
    #     normalizied_coordinate_vector = self.normalizer(distance_to_goal, min_value=0, max_value=self.floor_size)
        
    #     return normalizied_coordinate_vector

    def get_distance_to_goal(self):
        # """
        # Calculates and returns the normalized distances from the robot's current position to the four goals.

        # Returns:
        # - numpy.ndarray: Normalized distance vector.
        # """

        gps_value = self.gps.getValues()[0:2]
        current_coordinate = np.array(gps_value)

        destination_coordinates = np.array([[30, -30], [-30, 30], [30, 30], [-30, -30]])
        distances_to_goals = np.linalg.norm(destination_coordinates - current_coordinate, axis=1)
        normalized_distances = self.normalizer(distances_to_goals, min_value=0, max_value=self.floor_size)

        return normalized_distances

    # def get_distance_to_start(self):
    #     """
    #     Calculates and returns the normalized distance from the robot's current position to the goal.
        
    #     Returns:
    #     - numpy.ndarray: Normalized distance vector.
    #     """
        
    #     gps_value = self.gps.getValues()[0:2]
    #     current_coordinate = np.array(gps_value)
    #     distance_to_start = np.linalg.norm(self.start_coordinate - current_coordinate)
    #     normalizied_coordinate_vector = self.normalizer(distance_to_start, min_value=0, max_value=self.floor_size)
        
    #     return normalizied_coordinate_vector
    

    # def get_sensor_data(self):
    #     """
    #     Retrieves and normalizes data from distance sensors.
        
    #     Returns:
    #     - numpy.ndarray: Normalized distance sensor data.
    #     """
        
    #     # Gather values of distance sensors.
    #     sensor_data = []
    #     for z in self.dist_sensors:
    #         sensor_data.append(self.dist_sensors[z].value)  
            
    #     sensor_data = np.array(sensor_data)
    #     normalized_sensor_data = self.normalizer(sensor_data, self.min_sensor, self.max_sensor)
        
    #     return normalized_sensor_data
        
    def send(self , observation):
        message = struct.pack("f f f f f f f", observation[0] , observation[1], observation[2] , observation[3], observation[4] , observation[5], observation[6]) # Pack the message.
        
        self.emitter.send(message) # Send out the message
        # robot.step(5)    


    def receive(self):

        if self.receiver.getQueueLength() > 0: # If receiver queue is not empty
            receivedData = self.receiver.getBytes()
            tup = struct.unpack("f f f f f f f", receivedData) # Parse data into char, float, int
            self.receiver.nextPacket()
        else : 
            tup = [0 for i in range(7)]
        return tup
    
    def get_current_position(self):
        """
        Retrieves and normalizes data from distance sensors.
        
        Returns:
        - numpy.ndarray: Normalized distance sensor data.
        """
        
        # Gather values of distance sensors.
    
        position = self.gps.getValues()[0:3]
        position = np.array(position)

        roll = self.imu.getRollPitchYaw()[0]
        pitch = self.imu.getRollPitchYaw()[1]
        yaw = self.imu.getRollPitchYaw()[2]

        roll_velocity = self.gyro.getValues()[0]
        pitch_velocity = self.gyro.getValues()[1]
        yaw_velocity = self.gyro.getValues()[2]

        north0 = self.compass.getValues()[0]
        north1 = self.compass.getValues()[1]

        # normalized_current_position = self.normalizer(position, -4, +4)

        return position, roll, pitch, yaw, roll_velocity, pitch_velocity, yaw_velocity, north0, north1
    

    def get_observations(self):

        # """
        # Obtains and returns the normalized sensor data, current distance to the goal, and current position of the robot.
    
        # Returns:
        # - numpy.ndarray: State vector representing distance to goal, distance sensor values, and current position.
        # """

        # position, roll, pitch, yaw, north0, north1 = self.get_current_position()

        position, _, _, yaw, _, _, yaw_velocity, north0, north1 = self.get_current_position()

        # print(position[0])
        # print(position[1])
        # print(position[2])
        # print(yaw)
        # print(yaw_velocity)
        # print(north0)
        # print(north1)

        # Normal!
        observation = np.array([position[0], position[1], position[2], yaw, yaw_velocity, north0, north1])

        self.send(observation)

        recivedObservations = self.receive()

        observation = np.concatenate([observation , recivedObservations])
        # print(state_vector)

        return observation
    
    
    def apply_action(self, action):
        """
        Applies the specified action to the robot's motors.
        
        Returns:
        - None
        """
        position, roll, pitch, yaw, roll_velocity, pitch_velocity, yaw_velocity, north0, north1 = self.get_current_position()
        altitude = position[2]

        roll_disturbance = 0.0
        pitch_disturbance = 0.0
        yaw_disturbance = 0.0

        self.front_left_motor.setPosition(float('+inf'))
        self.front_right_motor.setPosition(float('+inf'))        
        self.rear_left_motor.setPosition(float('+inf'))
        self.rear_right_motor.setPosition(float('+inf'))
            
        # Transform the keyboard input to disturbances on the stabilization algorithm.
        roll_disturbance = 0.0
        pitch_disturbance = 0.0
        yaw_disturbance = 0.0

        if action == 0:
            pitch_disturbance = -2.0
        elif action == 1:
            pitch_disturbance = 2.0
        elif action == 2:
            roll_disturbance = -1.0
        elif action == 3:
            roll_disturbance = 1.0
        elif action ==4:
            if self.target_altitude >=1 and self.target_altitude<=5:
                self.target_altitude += 0.05
            # print("target altitude:", self.target_altitude, "[m]")
        elif action ==5 :
            if self.target_altitude<=1:
                pass
            else:
                self.target_altitude -= 0.05
            # print("target altitude:", self.target_altitude, "[m]")
        elif action ==6:
            roll_disturbance = 0.0
            pitch_disturbance = 0.0
            yaw_disturbance = 0.0


        # Compute the roll, pitch, yaw and vertical inputs.
        roll_input = self.k_roll_p * max(min(roll, 1.0), -1.0) + roll_velocity + roll_disturbance

        pitch_input = self.k_pitch_p * max(min(pitch, 1.0), -1.0) + pitch_velocity + pitch_disturbance
        yaw_input = yaw_disturbance
        clamped_difference_altitude = max(min(self.target_altitude - altitude + self.k_vertical_offset, 1.0), -1.0)
        vertical_input = self.k_vertical_p * math.pow(clamped_difference_altitude, 3.0)

        # Actuate the motors taking into consideration all the computed inputs.
        front_left_motor_input = self.k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input
        front_right_motor_input = self.k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input
        rear_left_motor_input = self.k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input
        rear_right_motor_input = self.k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input

        # Set the motor velocities.
        self.front_left_motor.setVelocity(front_left_motor_input)
        self.front_right_motor.setVelocity(-front_right_motor_input)
        self.rear_left_motor.setVelocity(-rear_left_motor_input)
        self.rear_right_motor.setVelocity(rear_right_motor_input)

        robot.step(5)        
        self.front_left_motor.setPosition(float(0))
        self.front_left_motor.setVelocity(68.8336792818065)
        self.front_right_motor.setPosition(float(0))
        self.front_right_motor.setVelocity(68.960245492582)
        self.rear_left_motor.setPosition(float(0))
        self.rear_left_motor.setVelocity(68.5356713468079)
        self.rear_right_motor.setPosition(float(0))
        self.rear_right_motor.setVelocity(68.66223755758341)
 
    

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state and returns the initial observations.
        
        Returns:
        - numpy.ndarray: Initial state vector.
        """

        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        return self.get_observations(), {}


    def step(self, action):    
        """
        Takes a step in the environment based on the given action.
        
        Returns:
        - state       = float numpy.ndarray with shape of (3,)
        - step_reward = float
        - done        = bool
        """
        self.apply_action(action)
        step_reward, done = self.get_reward()
        state = self.get_observations()
        # Time-based termination condition
        if (int(self.getTime()) + 1) % self.max_steps == 0:
            done = True
        none = 0
        return state, step_reward, done, none, {}
        
    
    def get_reward(self):
        """
        Calculates and returns the reward based on the current state.
        
        Returns:
        - The reward and done flag.
        """
        position, roll, pitch, yaw, roll_velocity, pitch_velocity, yaw_velocity, north0, north1 = self.get_current_position()
        altitude = position[2]

        done = False
        reward = 0

        if self.gyro.getValues()[0] > 11 or self.gyro.getValues()[1] > 11 or self.gyro.getValues()[2] > 11 :
            done =True
            reward -= 20
            print('reset')
        
        check_collision11 = self.touch11.value
        check_collision12 = self.touch12.value
        # check_collision21 = self.touch21.value
        # check_collision22 = self.touch22.value


        # print(self.gyro.getValues())
        if check_collision11 or check_collision12:
            # Punish if Collision
            done = True
            reward -= 5
            print('Collision!')
            
        if altitude < 0.1:
            reward -= 0.5
            
        if altitude > 0.2 and altitude<0.5:
            reward += 1
            
        if altitude > 0.5 and altitude < 1:
            reward += 2
            
        if altitude > 1 and altitude < 1.5:
            reward += 6

        if altitude > 1.5 and altitude < 2:
            reward += 8

        if altitude > 2:
            reward -= 0.5
            
        distance_to_goals = self.get_distance_to_goal()
        distance_to_goals *= 100 # The value is between 0 and 1. Multiply by 100 will make the function work better

        # print(distance_to_goals)

        min_distance = np.min(distance_to_goals)
        min_distance_index = np.argmin(distance_to_goals) 

        if min_distance < 30:
            if  min_distance < 4.5:
                growth_factor = 10
                A = 10
                done = True
                print("***************************************SOLVED***************************************")
            elif min_distance < 5:
                growth_factor = 7
                A = 15
            elif min_distance < 7:
                growth_factor = 7
                A = 10
            elif min_distance < 9:
                growth_factor = 7
                A = 7
            elif min_distance < 15:
                growth_factor = 3
                A = 3.5            
            elif min_distance < 20:
                growth_factor = 2
                A = 3.1
            elif min_distance < 25:
                growth_factor = 1.5
                A = 2.5
            elif min_distance < 28:
                growth_factor = 1
                A = 1.5
            else:
                growth_factor = 1
                A = 0.9

            reward += A * (1 - np.exp(-growth_factor * (1 / min_distance)))
        else: 
            reward += -min_distance / 100
        
        def cosine_similarity(vector1, vector2):
            dot_product = np.dot(vector1, vector2)
            magnitude1 = np.sqrt(np.dot(vector1, vector1))
            magnitude2 = np.sqrt(np.dot(vector2, vector2))
            cosine_similarity = dot_product / (magnitude1 * magnitude2)
            return cosine_similarity
        
        # ANGLE P/R
        goal_vectors = self.destination_coordinates - self.startinig_coordinates

        # print(f"goal vectors: {goal_vectors}")
        current_position = np.array([position[0], position[1]])  # Replace x and y with the actual coordinates of the quadcopter
        current_vectors = self.destination_coordinates - current_position

        # print(f"current_vectors: {current_vectors}")
        # print(self.destination_coordinates)
        # print(f"goal_vectors: {goal_vectors}")
        # print(f"current_position: {current_vectors}")
        # angle_similarities = cosine_similarity(goal_vectors, current_vectors)

        angle_list = [cosine_similarity(goal_vectors[0], current_vectors[0]), cosine_similarity(goal_vectors[1], current_vectors[1]),
                      cosine_similarity(goal_vectors[2], current_vectors[2]), cosine_similarity(goal_vectors[3], current_vectors[3])]


        angle = angle_list[min_distance_index]
        if angle > 0.9 and angle<0.93:
            reward += 2
        if angle > 0.93 and angle< 0.95:
            reward += 4
        if angle > 0.95 and angle < 0.98:
            reward += 6     
        if angle > 0.98:
            reward += 8
            
        if angle < 0.9 and angle > 0.85:
            reward -10
        if angle < 0.85 and angle > 0.8:
            reward -30


        

        return reward, done


class Agent_FUNCTION():
    def __init__(self, save_path, num_episodes):
        self.save_path = save_path
        self.num_episodes = num_episodes

        self.env = Environment()
        self.env = Monitor(self.env, "tmp/")

        #PPO
        self.policy_network = PPO("MlpPolicy", self.env, verbose=1, tensorboard_log="./results_tensorboard/")#.learn(total_timesteps=1000)
        
    
    def save(self):
        print(self.save_path ,"PPO-Best")
        self.policy_network.save(self.save_path + "PPO-Best")

    def load(self):
        self.policy_network = PPO.load(self.save_path + "CPPO-Best")


    def train(self) :

        log_dir = "tmp/"
        os.makedirs(log_dir, exist_ok=True)
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
        # Train the agent
        self.policy_network.learn(total_timesteps=int(self.num_episodes*100), callback=callback)

        self.save()

 
    def test(self):
        
        state = self.env.reset()
        for episode in range(1, self.num_episodes+1):
            rewards = []
            # state = self.env.reset()
            done = False
            ep_reward = 0
            state=np.array(state[0])
            while not done:
                # Ensure the state is in the correct format (convert to numpy array if needed)
                # print("state: ", state)
                state_np = np.array(state)

                # Get the action from the policy network
                action, _ = self.policy_network.predict(state_np)

                # Take the action in the environment
                state, reward, done, _,_ = self.env.step(action)
                # print("reward: ", reward)
                ep_reward += reward

            rewards.append(ep_reward)
            print(f"Episode {episode}: Score = {ep_reward:.3f}")
            state = self.env.reset()
class Agent_FUNCTION1(): #CTrain
    def __init__(self, save_path, num_episodes, env):
        self.save_path = save_path
        self.num_episodes = num_episodes
        self.env = env

    
    # def save(self):
    #     print(self.save_path ,"PPO-Best")
    #     self.policy_network.save(self.save_path + "PPO-Best")

    # def load(self):

    #     self.policy_network = PPO.load("./tmp/PPO-Best")
    def save(self):
        print(self.save_path ,"PPO-Best")
        self.policy_network.save(self.save_path + "CPPO-Best")

    def load(self):
        self.policy_network = PPO.load(self.save_path + "PPO-Best")



    def train(self) :

        log_dir = "tmp/"
        os.makedirs(log_dir, exist_ok=True)
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
        # Train the agent
        self.policy_network.learn(total_timesteps=int(self.num_episodes*100), callback=callback)

        self.save()

 
    def test(self):
        
        state = self.env.reset()
        for episode in range(1, self.num_episodes+1):
            rewards = []
            done = False
            ep_reward = 0
            state=np.array(state[0])
            while not done:
                # Ensure the state is in the correct format (convert to numpy array if needed)
                state_np = np.array(state)

                # Get the action from the policy network
                action, _ = self.policy_network.predict(state_np)

                # Take the action in the environment
                state, reward, done, _,_ = self.env.step(action)
                ep_reward += reward

            rewards.append(ep_reward)
            print(f"Episode {episode}: Score = {ep_reward:.3f}")
            state = self.env.reset()       
            
            
if __name__ == '__main__':
    # Configs
    save_path = './results/'   
    train_mode = "Test"  
    num_episodes = 500 if train_mode == "Train" or "CTrain" else 25

    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    


    if train_mode == "Train":
        # Initialize Training
        agent = Agent_FUNCTION(save_path, num_episodes)
        agent.train()
    elif train_mode == "Test":
        agent = Agent_FUNCTION(save_path, 20)

        # Load PPO
        agent.load()
        # Test
        agent.test()
    elif train_mode == "CTrain":
        env = Environment()
        agent = Agent_FUNCTION1(save_path, num_episodes, env)
        
        env = Monitor(env, "tmp/")
        # env = Environment()
        agent.load()
        agent.policy_network.set_env(env)
        agent.train()
