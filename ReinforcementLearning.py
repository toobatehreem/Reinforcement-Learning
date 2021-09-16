import numpy as np

R = np.matrix([[-1,-1,-1,-1,0,-1], #(0,4) available actions #row 0
        [-1,-1,-1,0,-1,100], #(1,3) and (1,5) available actions #row 1
        [-1,-1,-1,0,-1,-1], #row 2
        [-1,0, 0, -1, 0, -1], #row 3
        [-1,0,0,-1,-1,100], #row 4
        [-1,0,-1,-1,0,100]]) #row 5
print(R)

#Q matrix traversing among among 6 states (0-5)
Q = np.zeros([6,6])
print(Q)

#Gamma (Learning parameter
gamma = 0.8

#Initial state(random)
initial_state = 1

#functions returns all the available actions in the state given as arguments
def available_actions(state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1] #where in my row the state > or = 0
    return av_act #available actions

#get available actions in the current state
available_act = available_actions(initial_state)

#this function choose at random which action to be performed within the range of all the available actions
def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_act,1))
    return next_action

#sample next action to be performed
action = sample_next_action(available_act)


#this function updates the Q matrix according to the path selected and the Q learning algo
def update(current_state, action, gamma):
    max_index = np.where(Q[action,] == np.max(Q[action,]))[0]

    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size=1))
    else:
        max_index = int(max_index)
    max_value = Q[action, max_index]

#Q learning algo
    Q[current_state, action] = R[current_state, action] + gamma * max_value

#Update Q matrix
update(initial_state,action, gamma)

#trainig phase
#train over 10 000 iterations(re-iterate the above process)

for i in range(10000):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(current_state)
    action = sample_next_action(available_act)
    update(current_state, action,gamma)

#normalized the "trained"
#Q matrix
print('Trained Q matrix')
print(Q/np.max(Q)*100)


#testing
#goal state =  5
#calculating the best policy
current_state = 1
steps = [current_state]

while current_state !=5:
    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[0]
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size=1))
    else:
        next_step_index = int(next_step_index)

    steps.append(next_step_index)
    current_state = next_step_index

#printing delected sequence of steps
print('Selected path:')
print(steps)