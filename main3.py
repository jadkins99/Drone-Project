# import statements
import vrep
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import random as r
import csv

#test
# static functions not needed to be in a class
# function for encoding the state from the different sensor states. (for Q-Table)
def state_encoder(fs,rs,ls,bs,tl,td):
    # fs + 3rs + (3**2)ls + (3**3)bs + (3**4)tl + (3**4)*5*td
    state = (fs + 3*rs + (3**2)*ls + (3**3)*bs + (3**4)*tl + (3**4)*5*td)
    return state


# function for decoding the from state vector from the state number (Q-Table)
# state_vector = <ffs,rs,ls,bs,tl,td>
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


# function for getting the reward of a state/action pair also returns the dynamic and static variables
def reward_policy(object_levels, target_level, object_distances, old_object_distances, target_distance, old_target_distance):
    static = (np.min(object_levels) - 1) + (1/4)*(4 - target_level)
    dynamic = (2*(np.min(object_distances) - np.min(old_object_distances)) + (old_target_distance - target_distance))

    #print("Something is wrong")
    #print("obj dist diff ", np.min(object_distances) - np.min(old_object_distances))
    print("goal dist diff", (old_target_distance - target_distance))
    print("goal dist  ", target_distance)
    print("old goal dist ", old_target_distance)
    reward = 2*dynamic + 0.1*static
    return reward,static,dynamic


# function for normalizing the x and y components of our vector
def normalize(vec):
    max_val = math.sqrt(vec[0]**2+vec[1]**2)

    if max_val == 0:
        return vec

    else:
        vec[0] = vec[0]/max_val
        vec[1] = vec[1]/max_val
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
        # member variable for the targets of the drone
        self.returnCode, self.target1 = vrep.simxGetObjectHandle(clientID, "Goal1",
                                                                 vrep.simx_opmode_blocking)

        self.returnCode, self.target2 = vrep.simxGetObjectHandle(clientID, "Goal2",
                                                                 vrep.simx_opmode_blocking)

        self.returnCode, self.target3 = vrep.simxGetObjectHandle(clientID, "Goal3",
                                                                 vrep.simx_opmode_blocking)

        self.returnCode, self.target4 = vrep.simxGetObjectHandle(clientID, "Goal4",
                                                                 vrep.simx_opmode_blocking)
        self.new_target_bool = False
        # select the first target randomly
        self.chosen_target = r.randint(1,4)

        if self.chosen_target == 1:
            self.goal_handle = self.target1
        elif self.chosen_target == 2:
            self.goal_handle = self.target2
        elif self.chosen_target == 3:
            self.goal_handle = self.target3
        elif self.chosen_target == 4:
            self.goal_handle = self.target4

        # initialize member variable for getting the distance to the goal
        self.returnCode, self.goal_distM = vrep.simxGetObjectPosition(clientID, self.goal_handle, self.quad,
                                                                      vrep.simx_opmode_streaming)

        # initialize member variable for getting the angle from the drone to the goal
        self.angle = math.atan2(self.goal_distM[1], self.goal_distM[0])*(180/math.pi)

        # initialize variable for drone position
        self.returnCode, self.position = vrep.simxGetObjectPosition(clientID, self.quad, -1, vrep.simx_opmode_streaming)

        # initialize variables for sensors
        self.returnCode, self.detectionStateF, self.frontM, x2, x1 = \
            vrep.simxReadProximitySensor(clientID, self.front_sensor, vrep.simx_opmode_streaming)

        time.sleep(0.1)
        self.returnCode, self.detectionStateB, self.backM, x2, x1 = \
            vrep.simxReadProximitySensor(clientID, self.back_sensor, vrep.simx_opmode_streaming)

        time.sleep(0.1)
        self.returnCode, self.detectionStateL, self.leftM, x2, x1 = \
            vrep.simxReadProximitySensor(clientID, self.left_sensor, vrep.simx_opmode_streaming)

        time.sleep(0.1)
        self.returnCode, self.detectionStateR, self.rightM, x2, x1 = \
            vrep.simxReadProximitySensor(clientID, self.right_sensor, vrep.simx_opmode_streaming)

    # method for updating sensors
    def update_sensor(self):

        time.sleep(0.1)
        self.returnCode, self.detectionStateF, self.frontM, x2, x1 = \
            vrep.simxReadProximitySensor(self.clientID, self.front_sensor, vrep.simx_opmode_streaming)
        # if not detecting the object set the dist to 2.0
        if not self.detectionStateF:
            self.frontM = 2.0

        time.sleep(0.1)
        self.returnCode, self.detectionStateB, self.backM, x2, x1 = \
            vrep.simxReadProximitySensor(self.clientID, self.back_sensor, vrep.simx_opmode_streaming)
        if not self.detectionStateB:
            self.backM = 2.0

        time.sleep(0.1)
        self.returnCode, self.detectionStateL, self.leftM, x2, x1 = \
            vrep.simxReadProximitySensor(self.clientID, self.left_sensor, vrep.simx_opmode_streaming)
        if not self.detectionStateL:
            self.leftM = 2.0

        time.sleep(0.1)
        self.returnCode, self.detectionStateR, self.rightM, x2, x1 = \
            vrep.simxReadProximitySensor(self.clientID, self.right_sensor, vrep.simx_opmode_streaming)
        if not self.detectionStateR:
            self.rightM = 2.0

    # method for updating position
    def update_pos(self):
        self.position = self.returnCode, self.position = \
            vrep.simxGetObjectPosition(self.clientID, self.quad, -1, vrep.simx_opmode_streaming)


    # method for updating the goal distance
    def update_goal_dist(self):
        self.returnCode, self.goal_distM = \
            vrep.simxGetObjectPosition(self.clientID, self.goal_handle, self.quad, vrep.simx_opmode_streaming)

    # method for updating the current target of drone if closer than 3 meters, change the target
    def update_target(self):
        if np.linalg.norm(self.get_goal_dist()) < 4.0:
            new_target = r.randint(1, 4)
            # boolean for checking if the target has been changed
            self.new_target_bool = True
            # if the chosen target is the same as new target, recursively call to get a new target
            if self.chosen_target == new_target:
                self.update_target()
            # set chosen target to new target
            self.chosen_target = new_target
            # set the target handle to the new target
            if new_target == 1: self.goal_handle = self.target1
            elif new_target == 2: self.goal_handle = self.target2
            elif new_target == 3: self.goal_handle = self.target3
            elif new_target == 4: self.goal_handle = self.target4

        else:
            # boolean for checking if the target has been changed
            self.new_target_bool = False

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
        if self.get_front_state() == 2 and self.get_back_state() == 2 and self.get_right_state() == 2 and self.get_left_state() == 2:
            return False
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
            new_vec = np.array([x, y, z])
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
            object_dist_list = np.append(object_dist_list, np.linalg.norm(dist_vec))
        if object_dist_list.size == 0:
            return np.array([0, 0, 0])

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
            # if action is 0 but the sensor is 0 don't move the drone. else, move like normal
            if (self.get_front_state() == 0 or self.get_right_state() == 0 or self.get_back_state() == 0 or
                    self.get_left_state() == 0):
                self.set_target(0.0, 0.0, 3 - self.get_pos()[2])
            else:
                norm = normalize(self.get_closest_object())
                x = how_far*norm[0]
                y = how_far*norm[1]
                self.set_target(x, y, 3 - self.get_pos()[2])

        elif action == 1:
            norm = normalize(self.get_closest_object())

            time.sleep(2)
            # change vector by 90 degrees
            x = -how_far*norm[1]
            y = how_far*norm[0]
            self.set_target(x, y, 3 - self.get_pos()[2])

        elif action == 2:
            norm = normalize(self.get_closest_object())
            # change vector by -90 degrees
            x = how_far * norm[1]
            y = -1*how_far * norm[0]
            self.set_target(x, y, 3 - self.get_pos()[2])

        elif action == 3:
            norm = normalize(self.get_closest_object())
            # change vector by 180 degrees
            x = -1*how_far * norm[0]
            y = -1*how_far * norm[1]
            self.set_target(x, y, 3 - self.get_pos()[2])

        elif action == 4:
            # if the object is in the same direction of the target and the target is close, then dont move
            if self.get_goal_angle_state == 0 and self.get_front_state == 0:
                self.set_target(0.0,0.0,3 - self.get_pos()[2])

            elif self.get_goal_angle_state == 1 and self.get_left_state == 0:
                self.set_target(0.0,0.0,3 - self.get_pos()[2])

            elif self.get_goal_angle_state == 2 and self.get_back_state == 0:
                self.set_target(0.0,0.0,3 - self.get_pos()[2])

            elif self.get_goal_angle_state == 3 and self.get_right_state == 0:
                self.set_target(0.0,0.0,3 - self.get_pos()[2])

            else:
                norm = normalize(self.get_goal_dist())
                x = how_far*norm[0]
                y = how_far*norm[1]
                self.set_target(x, y, 3 - self.get_pos()[2])

        elif action == 5:
            norm = normalize(self.get_goal_dist())
            # change vector by 90 degrees
            x = -1*how_far * norm[1]
            y = how_far * norm[0]
            self.set_target(x, y, 3 - self.get_pos()[2])

        elif action == 6:
            # change vector by -90 degrees
            norm = normalize(self.get_goal_dist())
            x = how_far * norm[1]
            y = -1*how_far * norm[0]
            self.set_target(x, y, 3 - self.get_pos()[2])

        elif action == 7:
            # change vector by 180 degrees
            norm = normalize(self.get_goal_dist())
            x = -1*how_far * norm[0]
            y = -1*how_far * norm[1]
            self.set_target(x, y, 3 - self.get_pos()[2])


# class for the Q-Table
# action = 0,1,2,3 corresponds to towards, 90 left, 90 right, and 180 in direction of object
# action = 4,5,6,7 corresponds to towards, 90 left, 90 right, and 180 in direction of goal

class QTable:
    def __init__(self, states, actions, alpha, gamma):
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.QTable = np.zeros((states, actions))
        self.countTable = np.zeros((states, actions))

    def update_q(self, old_pos, action, reward, new_pos):

        q_new = (1 - self.alpha)*self.QTable[old_pos][action] + self.alpha*(reward + self.gamma*self.QTable[new_pos].max())
        self.QTable[old_pos][action] = q_new
        self.countTable[old_pos][action] += 1

    def display(self):
        print(self.QTable)

    def displayCountTable(self):
        print(self.countTable)


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
    alpha = 0.3
    gamma = 0.1
    epsilon = 0.25
    movement_distance = 0.2
    eGreedy = False
    k = 4
    # create object for drone
    ron = Drone(clientID)

    # create object for Q-Table
    Q1 = QTable(states, actions, alpha, gamma)


# create an array for the history of Q values
    history = np.zeros((20000, 23))

    possible_actions = np.array([])

    # run loop for controlling the drone
    for i in range(20000):


        ron.update_sensor()
        ron.update_pos()
        ron.update_target()
        ron.update_goal_dist()
        ron.update_goal_angle()

        # identify current state
        old_state = state_encoder(ron.get_front_state(), ron.get_right_state(), ron.get_left_state(),
                                  ron.get_back_state(), ron.get_goal_dist_state(), ron.get_goal_angle_state())
        # variable for detecting if an an object is detected in the new state
        old_state_obj_detect = ron.object_detected()

        # if exploration is e-greedy
        if eGreedy:
            # explore or exploit
            if e_greedy(epsilon):
                print("GREEDY")
                # exploit
                # if all actions are equal, select randomly
                if (Q1.QTable[old_state][0] == Q1.QTable[old_state][1] == Q1.QTable[old_state][2] ==
                        Q1.QTable[old_state][3] == Q1.QTable[old_state][4] == Q1.QTable[old_state][5] ==
                        Q1.QTable[old_state][6] == Q1.QTable[old_state][7]):

                    if ron.object_detected():
                        action = r.randint(0, 7)
                    else:
                        action = r.randint(4, 7)

                # else, select the max Q-value
                else:
                    if ron.object_detected():
                        action = Q1.QTable[old_state].argmax()
                    else:
                        possible_actions = np.array([-100, -100, -100, -100, Q1.QTable[old_state][4], Q1.QTable[old_state][5], Q1.QTable[old_state][6], Q1.QTable[old_state][7]])
                        action = possible_actions.argmax()
            else:
                print("RANDOM")
                # explore
                if ron.object_detected():
                    action = r.randint(0, 7)
                else:
                    action = r.randint(4, 7)
        # else do the other exploration policy
        else:

            # if all actions are equal, select randomly
            if (Q1.QTable[old_state][0] == Q1.QTable[old_state][1] == Q1.QTable[old_state][2] ==
                    Q1.QTable[old_state][3] == Q1.QTable[old_state][4] == Q1.QTable[old_state][5] ==
                    Q1.QTable[old_state][6] == Q1.QTable[old_state][7]):

                if ron.object_detected():
                    action = r.randint(0, 7)
                else:
                    action = r.randint(4, 7)

            else:

                # if an object is not detected, set value of actions 0-3 so that they wont be chosen
                # copy the possible actions into an  array
                possible_actions = np.copy(Q1.QTable[old_state])
                if not ron.object_detected():
                    possible_actions[0] = -1000
                    possible_actions[1] = -1000
                    possible_actions[2] = -1000
                    possible_actions[3] = -1000
                # for each possible action, adjust for the number of times it has been used
                for j in range(0, len(possible_actions)):
                    possible_actions[j] += k/(1 + Q1.countTable[old_state][j])
                    # add the exploration value to the history for recording
                    history[i][j + 15] = possible_actions[j]
                # pick the max action
                action = possible_actions.argmax()

        # store old distances
        old_goal_dist = np.linalg.norm(ron.get_goal_dist())

        old_object_dist_vec = np.array([np.linalg.norm(ron.get_front_sense()), np.linalg.norm(ron.get_left_sense()),
                                        np.linalg.norm(ron.get_back_sense()), np.linalg.norm(ron.get_right_sense())])

        # move drone based on action
        ron.move_drone(action, movement_distance)
        time.sleep(1)

        # update sensors and identify new state

        ron.update_sensor()
        ron.update_pos()
        ron.update_goal_dist()
        ron.update_goal_angle()
        ron.update_target()

        # variable for new goal dist
        # sometimes new_goal_dist returns 0.0 which really messes up dynamic. if this happens, set to old goal dist
        new_goal_dist = np.linalg.norm(ron.get_goal_dist())

        if new_goal_dist == 0.0:
            new_goal_dist = old_goal_dist


        if old_goal_dist == 0.0:
            old_goal_dist = new_goal_dist


        # create vector of object state levels and object distances
        obj_lev_vec = np.array([ron.get_front_state(), ron.get_left_state(), ron.get_back_state(),
                                ron.get_right_state()])
        obj_dist_vec = np.array([np.linalg.norm(ron.get_front_sense()), np.linalg.norm(ron.get_left_sense()),
                                 np.linalg.norm(ron.get_back_sense()), np.linalg.norm(ron.get_right_sense())])

        # assess new state
        new_state = state_encoder(ron.get_front_state(), ron.get_right_state(), ron.get_left_state(),
                                  ron.get_back_state(), ron.get_goal_dist_state(), ron.get_goal_angle_state())

        # variable for detecting if an an object is detected in the new state
        new_state_object_detected = ron.object_detected()

        # calculate the reward
        # if the action is 0 and the object state is 0 then set a negative reward. else, be normal
        if action == 0 and (ron.get_front_state() == 0 or ron.get_right_state() == 0 or ron.get_back_state() == 0 or ron.get_left_state() == 0):
            reward = -2.0
            static = -2.0
            dynamic = 0.0
        # if the action is 4 and the obstacle is in the way of the target
        elif action == 4 and (ron.get_front_state() == 0 and ron.get_goal_angle_state() == 0):
            reward = -2.0
            static = -2.0
            dynamic = 0.0

        elif action == 4 and (ron.get_left_state() == 0 and ron.get_goal_angle_state() == 1):
            reward = -2.0
            static = -2.0
            dynamic = 0.0

        elif action == 4 and (ron.get_back_state() == 0 and ron.get_goal_angle_state() == 2):
            reward = -2.0
            static = -2.0
            dynamic = 0.0

        elif action == 4 and (ron.get_right_state() == 0 and ron.get_goal_angle_state() == 3):
            reward = -2.0
            static = -2.0
            dynamic = 0.0

        else:
            reward, static, dynamic = reward_policy(obj_lev_vec, ron.get_goal_dist_state(), obj_dist_vec,
                                                    old_object_dist_vec, new_goal_dist,
                                                    old_goal_dist)
            assert (min(obj_dist_vec) > 0.0)
            assert (min(old_object_dist_vec) > 0.0)


        # step 4 update Q-Table based reward policy

        if (old_state_obj_detect == False and new_state_object_detected == True):
            print("Skipped Q-Update")

        elif not ron.new_target_bool:

            assert (dynamic < 2.0)
            assert (dynamic > -2.0)
            Q1.update_q(old_state, action, reward, new_state)


        # add the old state, action, static, dynamic, reward, Q-values, and the count of state/actions pairs
        # to the history array
        history[i][0] = i
        history[i][1] = old_state
        history[i][2] = action
        history[i][3] = Q1.countTable[old_state][action]
        history[i][4] = static
        history[i][5] = dynamic
        history[i][6] = reward
        history[i][7] = Q1.QTable[old_state][0]
        history[i][8] = Q1.QTable[old_state][1]
        history[i][9] = Q1.QTable[old_state][2]
        history[i][10] = Q1.QTable[old_state][3]
        history[i][11] = Q1.QTable[old_state][4]
        history[i][12] = Q1.QTable[old_state][5]
        history[i][13] = Q1.QTable[old_state][6]
        history[i][14] = Q1.QTable[old_state][7]



        print("Current iteration = ", i)
        print("Action: ", action)
        print("old Dist: ", old_goal_dist)
        print("New Dist: ", np.linalg.norm(ron.get_goal_dist()))
        print("Old State = ", old_state, ", New State = ", new_state)
        print("Static: ", static)
        print("Dynamic: ", dynamic)
        print("Reward: ", reward)
        print("Q-Values: ", Q1.QTable[old_state])
        print("Target: ", ron.chosen_target)
        print("Exploration values: ", possible_actions)
        print("Old obj Dist Vec: ", old_object_dist_vec)
        print("New Obj Dist Vec: ", obj_dist_vec)
        print()



    with open("history3.csv",'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(history)

    writeFile.close()


    with open("Qtable.csv",'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(Q1.QTable)

    writeFile.close()


# run main
if __name__ == "__main__":
    main()

else:
    print("main() not running.")

