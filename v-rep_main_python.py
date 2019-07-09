
import vrep
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import random as r


# static functions not needed to be in a class
# function for encoding the state from the different sensor states. (for Q-Table)
def state_encoder(fs,rs,ls,bs,tl,td):
    # fs + 3rs + (3**2)ls + (3**3)bs + (3**4)tl + (3**4)*5*td
    state = (fs + 3*rs + (3**2)*ls + (3**3)*bs + (3**4)*tl + (3**4)*5*td)
    return state


# function for decoding the from state vector from the state number (Q-Table)
def state_decoder(state):
    state_vector = []
    fs = state % 3
    state_vector.append(fs)
    state //= 3
    rs = state % 3
    state_vector.append(rs)
    state //= 3
    ls = state % 3
    state_vector.append(ls)
    state //= 3
    bs = state % 3
    state_vector.append(bs)
    state //= 3
    tl = state % 5
    state_vector.append(tl)
    td = state//5
    state_vector.append(td)
    return state_vector


# epsilon greedy function for resolving exploitation vs exploration
def e_greedy(epsilon):
    n = r.random()
    if n > epsilon:
        return True
    else:
        return False


# function for getting the reward of a state/action pair
def reward_policy(object_levels,target_level,object_distances,old_object_distances,target_distance,old_target_distance):
    static = (np.min(object_levels) - 2) + (1/5)*(5 - target_level)
    dynamic = (np.min(object_distances) - np.min(old_object_distances) + (old_target_distance - target_distance))
    reward = static + dynamic
    return reward


def normalize(vec):

    if (float(vec[0])**2.0 + float(vec[1])**2.0) == 0:
       return vec

    else:
        vec[0] = vec[0]/(math.sqrt(vec[0]**2+vec[1]**2))
        vec[1] = vec[1]/(math.sqrt(vec[0]**2+vec[1]**2))
        return vec


# class for controlling drone
class Drone:

    def __init__(self, clientID):
        self.clientID = clientID
        # member variable for drone
        self.returnCode, self.quad = vrep.simxGetObjectHandle(clientID, "Quadricopter_base",
                                                                       vrep.simx_opmode_blocking)
        # member variable for the target of the drone (for moving the drone)
        self.returnCode, self.target_handle = vrep.simxGetObjectHandle(clientID,"Quadricopter_target",
                                                                       vrep.simx_opmode_blocking)
        # member variables for sensors
        self.returnCode, self.front_sensor = vrep.simxGetObjectHandle(clientID, "Quad_front_sensor",
                                                                      vrep.simx_opmode_blocking)
        self.returnCode, self.back_sensor = vrep.simxGetObjectHandle(clientID, "Quad_back_sensor",
                                                                     vrep.simx_opmode_blocking)
        self.returnCode, self.left_sensor = vrep.simxGetObjectHandle(clientID, "Quad_left_sensor",
                                                                     vrep.simx_opmode_blocking)
        self.returnCode, self.right_sensor = vrep.simxGetObjectHandle(clientID, "Quad_right_sensor",
                                                                      vrep.simx_opmode_blocking)
        # member variable for the handle of the goal target for the drone
        self.returnCode, self.goal_handle = vrep.simxGetObjectHandle(clientID,"Goal_Target",vrep.simx_opmode_blocking)

        # initialize member variable for getting the distance to the goal
        self.returnCode, self.goal_distM = vrep.simxGetObjectPosition(clientID, self.goal_handle, self.quad, vrep.simx_opmode_streaming)

        # initialize member variable for getting the angle from the drone to the goal
        self.angle = math.atan2(self.goal_distM[1], self.goal_distM[0])*(180/math.pi)

        # initialize variable for drone position
        self.returnCode, self.position = vrep.simxGetObjectPosition(clientID, self.quad, -1, vrep.simx_opmode_streaming)

        # initialize variables for sensors
        self.returnCode, self.detectionStateF, self.frontM, x2, x1 = vrep.simxReadProximitySensor(clientID,
                                                                                                  self.front_sensor, vrep.simx_opmode_streaming)

        time.sleep(0.1)
        self.returnCode, self.detectionStateB, self.backM, x2, x1 = vrep.simxReadProximitySensor(clientID, self.back_sensor,
                                                                                                 vrep.simx_opmode_streaming)

        time.sleep(0.1)
        self.returnCode, self.detectionStateL, self.leftM, x2, x1 = vrep.simxReadProximitySensor(clientID,
                                                                                                 self.left_sensor, vrep.simx_opmode_streaming)

        time.sleep(0.1)
        self.returnCode, self.detectionStateR, self.rightM, x2, x1 = vrep.simxReadProximitySensor(clientID,
                                                                                                  self.right_sensor,
                                                                                             vrep.simx_opmode_streaming)

    # method for updating sensors
    def update_sensor(self):

        time.sleep(0.1)
        self.returnCode, self.detectionStateF, self.frontM, x2, x1 = vrep.simxReadProximitySensor(self.clientID, self.front_sensor, vrep.simx_opmode_buffer)
        time.sleep(0.1)
        self.returnCode, self.detectionStateB, self.backM, x2, x1 = vrep.simxReadProximitySensor(self.clientID, self.back_sensor, vrep.simx_opmode_buffer)

        time.sleep(0.1)
        self.returnCode, self.detectionStateL, self.leftM, x2, x1 = vrep.simxReadProximitySensor(self.clientID, self.left_sensor, vrep.simx_opmode_buffer)

        time.sleep(0.1)
        self.returnCode, self.detectionStateR, self.rightM, x2, x1 = vrep.simxReadProximitySensor(self.clientID, self.right_sensor, vrep.simx_opmode_buffer)

    # method for updating position
    def update_pos(self):
        self.position = self.returnCode, self.position = vrep.simxGetObjectPosition(self.clientID, self.quad, -1, vrep.simx_opmode_buffer)

    # method for updating the goal distance
    def update_goal_dist(self):
        self.returnCode, self.goal_distM = vrep.simxGetObjectPosition(self.clientID, self.goal_handle, self.quad, vrep.simx_opmode_buffer)

    # method for updating the goal angle
    def update_goal_angle(self):
        self.angle = math.atan2(self.goal_distM[1], self.goal_distM[0])*(180/math.pi)

# methods for getting the sensor measurements and the states of the sensors
    def get_right_sense(self):
        return self.rightM

    def get_right_state(self):
        length = np.linalg.norm(self.rightM)
        if not self.detectionStateR:
            return 2
        else:
            if 0 <= length < 1:
                return 0
            elif 1 <= length < 2:
                return 1
            elif length >= 2:
                return 2

    def get_left_sense(self):
        return self.leftM

    def get_left_state(self):
        length = np.linalg.norm(self.leftM)
        if not self.detectionStateL:
            return 2
        else:
            if 0 <= length < 1:
                return 0
            elif 1 <= length < 2:
                return 1
            elif length >= 2:
                return 2

    def get_front_sense(self):
        return self.frontM

    def get_front_state(self):
        length = np.linalg.norm(self.frontM)
        if not self.detectionStateF:
            return 2
        else:
            if 0 <= length < 1:
                return 0
            elif 1 <= length < 2:
                return 1
            elif length >= 2:
                return 2

    def get_back_sense(self):
        return self.backM

    def get_back_state(self):
        length = np.linalg.norm(self.backM)
        if not self.detectionStateB:
            return 2
        else:
            if 0 <= length < 1:
                return 0
            elif 1 <= length < 2:
                return 1
            elif length >= 2:
                return 2

    # method for getting drone position
    def get_pos(self):
        return self.position

    # method for getting the distance to the target goal
    def get_goal_dist(self):
        return self.goal_distM

    # method for getting the state of the distance from the goal
    def get_goal_dist_state(self):
        length = np.linalg.norm(self.goal_distM)
        if 0 <= length < 2:
            return 0
        elif 2 <= length < 4:
            return 1
        elif 4 <= length < 6:
            return 2
        elif 6 <= length < 8:
            return 3
        elif length >= 8:
            return 4

    # method for getting the angle from the drone to the goal
    def get_goal_angle(self):
        return self.angle

    # Method for returning the state of an angle. Returns 0-3 for to give the direction of the goal.
    # 0 = front sensor, 1 = left sensor, 2 = back sensor, 3 = right sensor
    def get_goal_angle_state(self):
        if -45 < self.angle <= 45:
            return 0
        elif 45 < self.angle <= 135:
            return 1
        elif 135 < self.angle <= 180 or -180 <= self.angle <= -135:
            return 2
        elif -135 < self.angle <= -45:
            return 3

    # method for detecting if an object is in sensors
    def object_detected(self):
        if self.detectionStateF or self.detectionStateB or self.detectionStateL or self.detectionStateR:
            return True

    # method for returning the closest object
    def get_closest_object(self):
        object_list = []
        object_dist_list = np.array([])

        # add all detected object vectors to the list
        # change coordinate system from reference of the sensor to reference of the drone
        # front sensor
        if self.detectionStateF:
            front_vec = self.get_front_sense()
            x = front_vec[2]; y = front_vec[0]; z = front_vec[1]
            new_vec = np.array([x,y,z])
            object_list.append(new_vec)
        # back sensor
        if self.detectionStateB:
            back_vec = self.get_back_sense()

            x = -1*back_vec[2]; y = -1*back_vec[0]; z = back_vec[1]
            new_vec = np.array([x, y, z])
            object_list.append(new_vec)

        # right vector
        if self.detectionStateR:
            right_vec = self.get_right_sense()
            x = right_vec[0]; y = -1*right_vec[2]; z = right_vec[1]
            new_vec = np.array([x, y, z])
            object_list.append(new_vec)
        # left vector
        if self.detectionStateL:
            left_vec = self.get_left_sense()
            x = left_vec[0]; y = left_vec[2]; z = left_vec[1]
            new_vec = np.array([x, y, z])
            object_list.append(new_vec)

        # for each vector in the list, append the distance to the distance array
        for dist_vec in object_list:
            object_dist_list = np.append(object_dist_list,np.linalg.norm(dist_vec))
        if object_dist_list.size == 0:
            return np.array([0,0,0])
        else:
            # find the index of the min distance
            min_index = np.argmin(object_dist_list)

            return object_list[min_index]

    # method for setting drone target (for moving drone)
    def set_target(self, x, y, z):
        self.returnCode = vrep.simxSetObjectPosition(self.clientID, self.target_handle, self.quad, (x, y, z),
                                                         vrep.simx_opmode_oneshot)

    # method for moving the drone
    # action = 0,1,2,3 corresponds to towards, 90 left, 90 right, and 180 in direction of object
    # action = 4,5,6,7 corresponds to towards, 90 left, 90 right, and 180 in direction of goal
    def move_drone(self,action,how_far):

        if action == 0:
            norm = normalize(self.get_closest_object())
            x = how_far*norm[0]
            y = how_far*norm[1]
            self.set_target(x, y, 0)

        elif action == 1:
            norm = normalize(self.get_closest_object())

            time.sleep(2)
            # change vector by 90 degrees
            x = -how_far*norm[1]
            y = how_far*norm[0]
            self.set_target(x, y, 0)

        elif action == 2:
            norm = normalize(self.get_closest_object())
            # change vector by -90 degrees
            x = how_far * norm[1]
            y = -1*how_far * norm[0]
            self.set_target(x, y, 0)

        elif action == 3:
            norm = normalize(self.get_closest_object())
            # change vector by 180 degrees
            x = -1*how_far * norm[0]
            y = -1*how_far * norm[1]
            self.set_target(x, y, 0)

        elif action == 4:
            norm = normalize(self.get_goal_dist())
            x = how_far*norm[0]
            y = how_far*norm[1]
            self.set_target(x, y, 0)

        elif action == 5:
            norm = normalize(self.get_goal_dist())
            # change vector by 90 degrees
            x = -1*how_far * norm[1]
            y = how_far * norm[0]
            self.set_target(x, y, 0)

        elif action == 6:
            # change vector by -90 degrees
            norm = normalize(self.get_goal_dist())
            x = how_far * norm[1]
            y = -1*how_far * norm[0]
            self.set_target(x, y, 0)

        elif action == 7:
            # change vector by 180 degrees
            norm = normalize(self.get_goal_dist())
            x = -1*how_far * norm[0]
            y = -1*how_far * norm[1]
            self.set_target(x, y, 0)



# class for the Q-Table
# action = 0,1,2,3 corresponds to towards, 90 left, 90 right, and 180 in direction of object
# action = 4,5,6,7 corresponds to towards, 90 left, 90 right, and 180 in direction of goal

class QTable:
    def __init__(self, states, actions, alpha, gamma):
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.QTable = np.zeros((states,actions))

    def update_q(self,old_pos,action,reward,new_pos):
        q_new = (1 - self.alpha)*self.QTable[old_pos][action] + self.alpha*(reward + self.gamma*self.QTable[new_pos].max())
        self.QTable[old_pos][action] = q_new

    def display(self):
        print(self.QTable)




def main():

    # initialize v-rep
    vrep.simxFinish(-1)  # just in case, close all opened connections
    clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to V-REP
    if clientID != -1:
        print('Connected to remote API server')

    else:
        print("Connection not successful")
        sys.exit("Could not connect")

    # Setup
    # 3*3*3*3*5*4 = 1620, 8 actions
    states = 1620
    actions = 8
    alpha = 0.5
    gamma = 0.8
    epsilon = 0.9
    movement_distance = 0.1

    # create object for drone
    ron = Drone(clientID)

    # create object for Q-Table
    Q1 = QTable(states, actions, alpha, gamma)

    # Step 1 take in information and identify status
    ron.update_sensor()
    ron.update_sensor()
    ron.update_pos()
    ron.update_goal_dist()
    ron.update_goal_angle()

    # for i in range(2000):
    while True:

        # identify current state
        old_state = state_encoder(ron.get_front_state(), ron.get_right_state(), ron.get_left_state(),
                                      ron.get_back_state(), ron.get_goal_dist_state(), ron.get_goal_angle_state())
        # explore or exploit
        if e_greedy(epsilon):
            # exploit
            # if all actions are equal, select randomly
            if (Q1.QTable[old_state][0] == Q1.QTable[old_state][1] == Q1.QTable[old_state][2] ==
                    Q1.QTable[old_state][3] == Q1.QTable[old_state][4] == Q1.QTable[old_state][5] ==
                    Q1.QTable[old_state][6] == Q1.QTable[old_state][7]):

                action = r.randint(0, 7)
            # else, select the max Q-value
            else:
                action = Q1.QTable[old_state].argmax()

        else:
            # explore
            action = r.randint(0, 7)

        # Step 2 move drone based on action
        ron.move_drone(action, movement_distance)


        # step 3 update sensors and identify new state

        # store old distances
        old_goal_dist = np.linalg.norm(ron.get_goal_dist())
        old_object_dist_vec = np.array([np.linalg.norm(ron.get_front_sense()), np.linalg.norm(ron.get_left_sense()),
                                        np.linalg.norm(ron.get_back_sense()), np.linalg.norm(ron.get_right_sense())])

        # update sensors
        ron.update_sensor()
        ron.update_pos()
        ron.update_goal_dist()
        ron.update_goal_angle()

        # create vector of object state levels and object distances
        obj_lev_vec = np.array([ron.get_front_state(),ron.get_left_state(),ron.get_back_state(),ron.get_right_state()])
        obj_dist_vec = np.array([np.linalg.norm(ron.get_front_sense()),np.linalg.norm(ron.get_left_sense()),
                                 np.linalg.norm(ron.get_back_sense()),np.linalg.norm(ron.get_right_sense())])

        # assess new state
        new_state = state_encoder(ron.get_front_state(), ron.get_right_state(), ron.get_left_state(),
                                  ron.get_back_state(), ron.get_goal_dist_state(), ron.get_goal_angle_state())


        # calculate the reward
        reward = reward_policy(obj_lev_vec, ron.get_goal_dist_state(), obj_dist_vec, old_object_dist_vec, np.linalg.norm(ron.get_goal_dist()), old_goal_dist)

        # step 4 update Q-Table based reward policy

        Q1.update_q(old_state,action,reward,new_state)

        time.sleep(1)


# run main
if __name__ == "__main__":
    main()

else:
    print("main() not running.")

