import argparse
import os
import time
# from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator


parser = argparse.ArgumentParser(description='Configure the files before training the net.')
parser.add_argument('--output_file_name', default="rewards", type=str, help='Output Filename to store the rewards.')
parser.add_argument('--id_gpu', default=1, type=int, help='which gpu to use.')
# parser.add_argument('--data_folder', help='Location of the data directory', type=str)
parser.add_argument('--episodes', default=10, type=int, help='how many episodes to run.')
parser.add_argument('--epochs', default=5, type=int, help='how many epochs to run for.')
parser.add_argument('--steps', default=50, type=int, help='how many steps to run for.')
# parser.add_argument('--output_file_name', default="rewards", type=str, help='Output Filename to store the rewards.')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)

batch_size = 32


import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
#from ros_gz_interfaces.msg import Contacts
import matplotlib.pyplot as plt
import matplotlib.colors
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import cv2
# from sensor_msgs.msg import LaserScan as L1
# from sensor_msgs.msg import LaserScan as L2
from sensor_msgs.msg import LaserScan


from nav_msgs.msg import Odometry
import math


import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Input
import numpy as np
import random
import csv
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
import ignition
from tf2_msgs.msg import TFMessage
# print(ignition.msgs5)



print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.config.list_physical_devices('GPU')
print(device_lib.list_local_devices())






class VelocityPublisher(Node):

    def __init__(self, state_size, action_size, end_x1, end_y1, end_x2, end_y2):
        super().__init__('velocity_publisher')
        self.velocity_publisher1 = self.create_publisher(Twist, '/robot1/cmd_vel', 10)
        self.velocity_publisher2 = self.create_publisher(Twist, '/robot2/cmd_vel', 10)
        # self.velocity_publisher3 = self.create_publisher(Twist, '/robot3/cmd_vel', 10)

        timer_period = 0.5
        self.x1 = 0.0
        self.z1 = 0.0
        self.x2 = 0.0
        self.z2 = 0.0

        self.i = 0

        # self.rate = self.create_rate(10)
        self.i = 0

        self.lidar_distance1 = 0
        self.lidar_distance2 = 0
        # self.lidar_distance3 = 0

        self.state_size = state_size
        self.action_size = action_size
        self.cur_x1 = 0
        self.cur_y1 = 0
        self.cur_x2 = 0
        self.cur_y2 = 0
        # self.cur_x3 = 0
        # self.cur_y3 = 0
        self.end_x1 = end_x1
        self.end_y1 = end_y1
        self.end_x2 = end_x2
        self.end_y2 = end_y2
        # self.end_x3 = end_x3
        # self.end_y3 = end_y3

        self.Q_next_pred = np.array([0,0,0, 0, 0, 0])

        self.Q_next_pred1 = np.array([0,0,0])
        self.Q_next_pred2 = np.array([0,0,0])

        self.bump1 = 0
        self.bump2 = 0

        self.memory1 = deque(maxlen=2000)
        # self.memory2 = deque(maxlen=2000)
        # self.memory3 = deque(maxlen=2000)

        self.gamma = 0.9
        self.epsilon = 0.2
        self.alpha = 0.01

        # if os.path.isfile(args.output_file_name + "_model1_.keras") and os.path.isfile(args.output_file_name + "_model2_.keras"):
        if os.path.isfile(args.output_file_name + "_model1_.keras"):
            print("Loading models...")
            self.model1 = tf.keras.models.load_model(args.output_file_name + "_model1_.keras")
            # self.model2 = tf.keras.models.load_model(args.output_file_name + "_model2_.keras")

        else:
            print("Creating models...")
            self.model1 = self.build_model()
            # self.model2 = self.build_model()
        # self.model3 = self.build_model()

        print("Model 1 summary = ", self.model1.summary())
        # print("Model 2 summary = ", self.model2.summary())
        self.rewards = []
        self.rewards1 = []
        self.rewards2 = []
        self.step = 0
        self.reached = 0
        self.actions = {0: 'move_forward', 1:'rotate_right', 2:'rotate_left'}

        self.velocity1 = Twist()
        self.velocity2 = Twist()
        # self.velocity3 = Twist()

        self.bridge = CvBridge()




        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.lidar_subscription1 = self.create_subscription(LaserScan, '/robot1/scan', self.lidar_callback1, 10)
        print("**#############*****: ", self.lidar_subscription1)
        self.lidar_subscription2 = self.create_subscription(LaserScan, '/robot2/scan', self.lidar_callback2, 10)
        print("**************: ", self.lidar_subscription2)

        # self.lidar_subscription3 = self.create_subscription(LaserScan, '/robot3/scan', self.lidar_callback, 10)

        self.odom_subscription1 = self.create_subscription(Odometry, '/robot1/odom', self.odom_callback1, 10)
        self.odom_subscription2 = self.create_subscription(Odometry, '/robot2/odom', self.odom_callback2, 10)
        # self.odom_subscription3 = self.create_subscription(Odometry, '/robot3/odom', self.odom_callback, 10)

        self.bumper_subscription1 = self.create_subscription(Contacts, '/robot1/bumper_contact', self.bumper_callback1, 10)
        self.bumper_subscription2 = self.create_subscription(Contacts, '/robot2/bumper_contact', self.bumper_callback2, 10)

        # self.lidar_subscription2
        # self.lidar_subscription1
        # self.lidar_subscription3

        # self.odom_subscription1
        # self.odom_subscription2
        # # self.odom_subscription3
        #
        # self.bumper_subscription1
        # self.bumper_subscription2

        self.start_time = time.time()





    def build_model(self):

        # Neural Network for building the DQN
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation = 'relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha))

        return model

    def memorize(self, memory, y, Qa):
        memory.append((y, Qa))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):

        print("Training neural network 1...")
        minibatch = random.sample(self.memory1, batch_size)
        for i, (y, Qa) in enumerate(minibatch):
            y = np.expand_dims(y, axis=0)

            with tf.device('/gpu:1'):
                self.model1.fit(Qa, y, verbose=0, epochs=args.epochs)

        # print("Training neural network 2...")
        # minibatch = random.sample(self.memory2, batch_size)
        # for i, (y, Qa) in enumerate(minibatch):
        #     with tf.device('/gpu:1'):
        #         self.model2.fit(Qa, y, verbose=0, epochs=args.epochs)

        # print("Training neural network3...")
        # minibatch = random.sample(self.memory3, batch_size)
        # for i, (y, Qa) in enumerate(minibatch):
        #     with tf.device('/gpu:1'):
        #         self.model3.fit(Qa, y, verbose=0, epochs=args.epochs)

    def cal_reward(self, next_x, next_y, tar_x, tar_y, lidar_distance):

        if tar_x - 3 <= next_x <= tar_x + 3 and tar_y - 3 <= next_y <= tar_y + 3:
            return - ((abs(tar_x - next_x) + abs(tar_y - next_y)) ** 2)

        if tar_x - 1 <= next_x <= tar_x + 1 and tar_y - 1 <= next_y <= tar_y + 1:
            return 1000

        # Calculating lidar component of the reward
        if lidar_distance > 6:
            d = 0
        elif 6 >= lidar_distance >= 2:
            d = 100
        elif lidar_distance < 2:
            d = 1000

        return - ((abs(tar_x - next_x) + abs(tar_y - next_y))  + d )
        # return - (((abs(tar_x - next_x) + abs(tar_y - next_y)) ** 2) )

    def get_flops(self, model, model_inputs) -> float:
        """
        Calculate FLOPS [GFLOPs] for a tf.keras.Model or tf.keras.Sequential model
        in inference mode. It uses tf.compat.v1.profiler under the hood.
        """
        # if not hasattr(model, "model"):
        #     raise wandb.Error("self.model must be set before using this method.")

        if not isinstance(
            model, (tf.keras.models.Sequential, tf.keras.models.Model)
        ):
            raise ValueError(
                "Calculating FLOPS is only supported for "
                "`tf.keras.Model` and `tf.keras.Sequential` instances."
            )

        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2_as_graph,
        )

        # Compute FLOPs for one sample
        # batch_size = 1
        # inputs = [
        #     tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype)
        #     for inp in model_inputs
        # ]

        # convert tf.keras model into frozen graph to count FLOPs about operations used at inference
        real_model = tf.function(model).get_concrete_function(model_inputs)
        frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

        # Calculate FLOPs with tf.profiler
        run_meta = tf.compat.v1.RunMetadata()
        opts = (
            tf.compat.v1.profiler.ProfileOptionBuilder(
                tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
            )
            .with_empty_output()
            .build()
        )

        flops = tf.compat.v1.profiler.profile(
            graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
        )

        tf.compat.v1.reset_default_graph()

        # convert to GFLOPs
        return (flops.total_float_ops )/2

    def timer_callback(self):

        # print("Inside timer callback, i = ", self.i)
        # self.tf_static_broadcaster.sendTransform(self.t)

        input_to_models = np.array([[self.cur_x1, self.cur_y1, self.lidar_distance1,
                                     self.cur_x2, self.cur_y2, self.lidar_distance2 ]])
                                     # self.cur_x3, self.cur_y3, self.lidar_distance3 ]])




        if self.i > 5:


            # print("self.i = ", self.i)
            if self.end_x1 - 1 <= self.cur_x1 <= self.end_x1 + 1 and self.end_y1 - 1 <= self.cur_y1 <= self.end_y1 + 1:
                if self.end_x2 - 1 <= self.cur_x2 <= self.end_x2 + 1 and self.end_y2 - 1 <= self.cur_y2 <= self.end_y2 + 1:
                    # if self.end_x3 - 1 <= self.cur_x3 <= self.end_x3 + 1 and self.end_y3 - 1 <= self.cur_y3 <= self.end_y3 + 1:

                    self.reached = 1
                    self.model1.save(args.output_file_name + "model1.keras")
                    # self.model2.save(args.output_file_name + "model2.keras")
                    # self.model3.save(args.output_file_name + "model3.keras")

            if self.end_x1 - 1 <= self.cur_x1 <= self.end_x1 + 1 and self.end_y1 - 1 <= self.cur_y1 <= self.end_y1 + 1:
                print("Robot 1 reached the target location! ")

            if self.end_x2 - 1 <= self.cur_x2 <= self.end_x2 + 1 and self.end_y2 - 1 <= self.cur_y2 <= self.end_y2 + 1:
                print("Robot 2 reached the target location! ")


            # Taking greedy or exploratory action based on the epsilon
            if np.random.uniform() > self.epsilon:

                predict_start_time = time.time()
                self.Q_next_pred = self.model1.predict(input_to_models, verbose=0)
                print("Predict time = ", time.time() - predict_start_time)

                # Q_next_pred1 = self.model1.predict(input_to_models, verbose=0)
                # Q_next_pred2 = self.model2.predict(input_to_models, verbose=0)
                self.Q_next_pred1 = self.Q_next_pred[0][:3]
                self.Q_next_pred2 = self.Q_next_pred[0][3:]

                action_number1 = np.argmax(self.Q_next_pred1)
                action_number2 = np.argmax(self.Q_next_pred2)




                # action_number3 = np.argmax(self.model3.predict(input_to_models, verbose=0))

                # action_number = np.argmax(self.model.predict(np.array([[self.cur_x, self.cur_y]]), verbose=0))


            else:
                action_number1 = random.randrange(3)
                action_number2 = random.randrange(3)
                # action_number3 = random.randrange(3)



            print("Current actions = ",
                  "Robot 1 = ", self.actions[action_number1],
                  ", Robot 2 = ", self.actions[action_number2] )
                  # "Robot 3 = ", self.actions[action_number3])

            # self.take_action(action_number1, action_number2, action_number3)
            self.take_action(action_number1, action_number2)

            reward1 = self.cal_reward(self.cur_x1, self.cur_y1, self.end_x1, self.end_y1, self.lidar_distance1)
            reward2 = self.cal_reward(self.cur_x2, self.cur_y2, self.end_x2, self.end_y2, self.lidar_distance2)
            # reward3 = self.cal_reward(self.cur_x3, self.cur_y3, self.end_x3, self.end_y3)

            # reward = reward1 + reward2 + reward3
            reward = reward1 + reward2

            # reward, next_x, next_y = next_state(cur_x, cur_y, action, end_x, end_y)
            print("Reward = ", reward)
            print("Reward1 = ", reward1)
            print("Reward2 = ", reward2)
            self.rewards.append(reward)
            self.rewards1.append(reward1)
            self.rewards2.append(reward2)

            # if (self.cur_x, self.cur_y) != (self.end_x, self.end_y):
            print("Current state after taking action of robot 1 = ", self.cur_x1, self.cur_y1)
            print("Current state after taking action of robot 2 = ", self.cur_x2, self.cur_y2)
            # print("end state = ", self.end_x, self.end_y)





            # Q_next_pred3 = self.model3.predict(input_to_models, verbose=0)

            # Q_next_pred = self.model.predict(np.array([[self.cur_x, self.cur_y]]), verbose=0)



            self.Q_next_pred1[np.argmax(self.Q_next_pred1)] = reward + self.gamma * np.max(self.Q_next_pred1)
            self.Q_next_pred2[np.argmax(self.Q_next_pred2)] = reward + self.gamma * np.max(self.Q_next_pred2)
            # Q_next_pred3[0][np.argmax(Q_next_pred3)] = reward + self.gamma * np.max(Q_next_pred3)


            # else:
            #     Q_next_pred = reward

            self.memorize(self.memory1, self.Q_next_pred, input_to_models)
            # self.memorize(self.memory2, self.Q_next_pred2, input_to_models)

            with open(str(args.episodes)+"_episode_data", 'a') as g:
                # using csv.writer method from CSV package
                write = csv.writer(g)

                write.writerow(["Robot1 ", self.Q_next_pred1, input_to_models])
                write.writerow(["Robot2 ", self.Q_next_pred2, input_to_models])

                g.close()




            # self.memorize(memory3, Q_next_pred3, input_to_models3)

            # self.memorize(Q_next_pred, np.array([[self.cur_x, self.cur_y]]))


            # cur_x, cur_y = next_x, next_y
            print("step: {}".format(self.step + 1))

            if len(self.memory1) > batch_size:
                self.replay(batch_size)

            self.step += 1
            if self.step == args.steps:
                print("Couldn't reach the target!")

                end_time = time.time() - self.start_time
                print("Total time taken = ", end_time)
                with open(args.output_file_name, 'a') as f:
                    # using csv.writer method from CSV package
                    write = csv.writer(f)

                    write.writerow(self.rewards)
                    f.close()

                plt.plot(self.rewards, label='rewards')
                plt.xlabel('Steps', fontsize=20)
                plt.ylabel('Rewards', fontsize=20)
                plt.title('Rewards vs Steps', fontsize=20)
                plt.legend(fontsize=15)
                plt.savefig(args.output_file_name + str(time.time()) + "_cumulative_rewards_" + ".png")

                plt.plot(self.rewards, label='rewards')
                plt.xlabel('Steps', fontsize=20)
                plt.ylabel('Rewards', fontsize=20)
                plt.title('Rewards vs Steps', fontsize=20)
                plt.legend(fontsize=15)
                plt.savefig(args.output_file_name + str(time.time()) + "rewards1" + ".png")

                plt.plot(self.rewards, label='rewards')
                plt.xlabel('Steps', fontsize=20)
                plt.ylabel('Rewards', fontsize=20)
                plt.title('Rewards vs Steps', fontsize=20)
                plt.legend(fontsize=15)
                plt.savefig(args.output_file_name + str(time.time()) + "rewards2" + ".png")



                self.model1.save(args.output_file_name + "_model1_"  + ".keras")
                # self.model2.save(args.output_file_name + "_model2_"  + ".keras")
                # self.model.save(args.output_file_name + "_model3_" + str(time.time()) + ".keras")

                print("Model flops = ", self.get_flops(self.model1, tf.constant(input_to_models)))

                exit(0)
            #
            # stepcounts.append(step)
            # all_rewards.append(rewards)

        if self.reached:
            # self.x1 = self.x2 = self.x3 = 0.0
            self.x1 = self.x2 = 0.0

            # self.y1 = self.y2 = self.y3 = 0.0
            self.x1 = self.x2 = 0.0

            self.model1.save(args.output_file_name + "_model1_" + ".keras")
            # self.model2.save(args.output_file_name + "_model2_" + ".keras")
            # self.model.save(args.output_file_name + "_model3_"+ str(time.time()) + ".keras")


            with open(args.output_file_name, 'a') as f:
                # using csv.writer method from CSV package
                write = csv.writer(f)

                write.writerow(self.rewards)
                f.close()

            print("Reached the goal!")
            plt.plot(self.rewards, label='rewards')
            plt.xlabel('Steps', fontsize=20)
            plt.ylabel('Rewards', fontsize=20)
            plt.title('Rewards vs Steps', fontsize=20)
            plt.legend(fontsize=15)
            plt.savefig(args.output_file_name + str(time.time()) +".png")

            plt.show(block = True)
            exit(0)

            # plt.pause(3)


            # executor.shutdown(timeout_sec = 0)
            # self.destroy_node()

            # return()

        self.velocity1.linear.x = self.x1
        self.velocity1.angular.z = self.z1
        self.velocity_publisher1.publish(self.velocity1)

        self.velocity2.linear.x = self.x2
        self.velocity2.angular.z = self.z2
        self.velocity_publisher2.publish(self.velocity2)

        # self.velocity3.linear.x = self.x3
        # self.velocity3.angular.z = self.z3
        # self.velocity_publisher3.publish(self.velocity3)
        # self.get_logger().info('Publishing linear velocity: "%f" and angular velocity %f' % (self.velocity.linear.x, self.velocity.angular.z))
        self.i += 1



    def odom_callback1(self, msg):
        self.cur_x1 = msg.pose.pose.position.x
        self.cur_y1 = msg.pose.pose.position.y
    #
    def odom_callback2(self, msg):
        self.cur_x2 = msg.pose.pose.position.x
        self.cur_y2 = msg.pose.pose.position.y

    # def odom_callback3(self, msg):
    #     self.cur_x3 = msg.pose.pose.position.x
    #     self.cur_y3 = msg.pose.pose.position.y


    def lidar_callback1(self, msg1):
        if msg1.ranges[160] > 12:
            self.lidar_distance1 = 12
        else:
           self.lidar_distance1 = msg1.ranges[160]

    def lidar_callback2(self, msg2):
        if msg2.ranges[160] > 12:
            self.lidar_distance2 = 12
        else:
            self.lidar_distance2 = msg2.ranges[160]

    # def lidar_callback3(self, msg):
    #     self.lidar_distance3 = msg.ranges[160]

    def bumper_callback1(self, msg):

        if msg.contacts != []:
            # self.bump1 = 1
            self.i = 0
            # for i in range(100):
            # self.timer_callback(-0.2, 1.0)
            # self.x = 0.0
            # self.z = 0.5
            self.x1 = 0.0
            self.z1 = 0.5
        #
        # if self.i > 9:
        #     self.x = 1.0
        #     self.z = 0.0
        #
        # self.i += 1

    def bumper_callback2(self, msg):

        if msg.contacts != []:
            # self.bump2 = 1
            self.i = 0
            # for i in range(100):
            # self.timer_callback(-0.2, 1.0)
            # self.x = 0.0
            # self.z = 0.5
            self.x2 = 0.0
            self.z2 = 0.5

        # if self.i > 9:
        #     self.x = 1.0
        #     self.z = 0.0
        #
        # self.i += 1


    def take_action(self, action_number1, action_number2):

        self.i = 0

        # Robot 1
        # move forward
        if action_number1 == 0:
            self.x1 = 0.5
            self.z1 = 0.0

        # rotate right
        elif action_number1 == 1:
            self.x1 = 0.0
            self.z1 = -0.5

        # rotate left
        elif action_number1 == 2:
            self.x1 = 0.0
            self.z1 = 0.5

        # Robot 2
        # move forward
        if action_number2 == 0:
            self.x2 = 0.5
            self.z2 = 0.0

        # rotate right
        elif action_number2 == 1:
            self.x2 = 0.0
            self.z2 = -0.5

        # rotate left
        elif action_number2 == 2:
            self.x2 = 0.0
            self.z2 = 0.5

        # Robot3
        # move forward
        # if action_number3 == 0:
        #     self.x = 0.5
        #     self.z = 0.0
        #
        # # rotate right
        # elif action_number3 == 1:
        #     self.x = 0.0
        #     self.z = -0.5
        #
        # # rotate left
        # elif action_number3 == 2:
        #     self.x = 0.0
        #     self.z = 0.5




def main():
    print('Hello from turtlebot4.')

    rclpy.init()

    state_size = 6
    action_size = 6

    agent = VelocityPublisher(state_size, action_size, 12, -5, -9, 6)
    agent.model1.save(args.output_file_name + "_model1_.keras")
    # agent.model2.save(args.output_file_name + "_model2_.keras")


    rclpy.spin(agent)




    with open(args.output_file_name, 'w') as f:

        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerow(agent.rewards)


    plt.plot(agent.rewards, label='rewards')
    plt.xlabel('Steps', fontsize=20)
    plt.ylabel('Rewards', fontsize=20)
    plt.title('Rewards vs Steps', fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig(args.output_file_name)
    plt.show()

    agent.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
