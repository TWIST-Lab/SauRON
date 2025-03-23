# SauRON: Smart Surveillance using Robotic Swarms with Optimized Networks

This codebase facilitates the simulation of the initial experiments done for the SuoRON framework.

## Introduction

This experiment introduces the novel framework using the multiple agent Reinforcement Learning to train the multiple robots for the Smart Surveillance. The python file MARL.py contains the introductory code to be run on the ROS2 and Gazebo platform simulating the multiple turtlebots4. The data folder contains the text files with the rewards generated at each episode. The Model folder contains the saved keras models after Deep Q-Learning training was completed.

## Setup and Prerequisites

### Prerequisites

Before running the MARL.py file make sure you have the following installed.

- Ubuntu OS
- Python3 - [Download](https://www.python.org/downloads/)
- ROS2 (Humble) - [Download](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
- Ignition Gazebo for turtlebots 4 - [Download](https://turtlebot.github.io/turtlebot4-user-manual/software/turtlebot4_simulator.html)

### Getting Started

Once you have successfully installed all the necessary dependencies, proceed to run the python file. For runnuing the python file and creating a ROS package, see the instructions below.

### Instructions

1. First create a ROS 2 package using the guidelines given in this ROS 2 documentation. - [ROS 2 package creation](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html)

2. Run a gazebo with two turtlebots with different namespaces as robot1 and robot2.

3. Then put the MARL.py file in the src folder of the package created.

4. Then run the MARL.py file.

## Cite

If you find this simulation framework useful in your research, please cite our work:

```Gaurav Singh, Sunday Amatare, and  Debashri Roy, "SauRON: Smart Surveillance using Robotic Swarms with Optimized Networks", IEEE INFOCOM WKSHOPS: NetRobiCS 2025, Available at SSRN: https://dx.doi.org/10.2139/ssrn.5143487```

## Acknowledgments

Thank you for choosing the SauRON framework. For any questions, feedback, or collaboration opportunities, please feel free to reach out.
