import pandas
import numpy as np
import gymnasium as gym

np.set_printoptions(edgeitems=30)
data = np.genfromtxt('data_better.csv', delimiter=',')[1:-1]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def one_hot(s1, s2, s3, s4):
    a = np.array([s1, s2, s3, s4])
    b = np.zeros((a.size, 10))
    b[np.arange(a.size), a] = 1
    return b.reshape(-1)

actions = np.array(data[:,4]).astype(int)
rewards = data[:,9]

do_sigmoid = True
if (do_sigmoid):
    data[:,0] = np.vectorize(sigmoid)(data[:,0])
    data[:,1] = np.vectorize(sigmoid)(data[:,1])
    data[:,2] = np.vectorize(sigmoid)(data[:,2])
    data[:,3] = np.vectorize(sigmoid)(data[:,3])

    data[:,5] = np.vectorize(sigmoid)(data[:,5])
    data[:,6] = np.vectorize(sigmoid)(data[:,6])
    data[:,7] = np.vectorize(sigmoid)(data[:,7])
    data[:,8] = np.vectorize(sigmoid)(data[:,8])

s1 = pandas.qcut(data[:,0], 10, labels=range(0,10))
s2 = pandas.qcut(data[:,1], 10, labels=range(0,10))
s3 = pandas.qcut(data[:,2], 10, labels=range(0,10))
s4 = pandas.qcut(data[:,3], 10, labels=range(0,10))
states_onehot = np.array([one_hot(s1[i], s2[i], s3[i], s4[i]) for i in range(len(s1))]) # rows x

s1_cat = pandas.qcut(data[:,0], 10)
s2_cat = pandas.qcut(data[:,1], 10)
s3_cat = pandas.qcut(data[:,2], 10)
s4_cat = pandas.qcut(data[:,3], 10)

nx1 = pandas.qcut(data[:,5], 10, labels=range(0,10))
nx2 = pandas.qcut(data[:,6], 10, labels=range(0,10))
nx3 = pandas.qcut(data[:,7], 10, labels=range(0,10))
nx4 = pandas.qcut(data[:,8], 10, labels=range(0,10))
nextstates_onehot = np.array([one_hot(nx1[i], nx2[i], nx3[i], nx4[i]) for i in range(len(s1))])

weights = np.random.uniform(-0.001, 0.001, (40,2))

def update(state, action, reward, next_state, weights, lr):
    pred_next = next_state.dot(weights) # evaluating Q, (n x 40) dot (40 x 2) gives (n x 2)
    max_v = np.max(pred_next) # n x 1 where each entry is max Q value of that row
    y = reward + 0.99*max_v # n x 1
    temp = state.dot(weights) # n x 2
    q = np.array([temp[i][action[i]] for i in range(len(action))]) # n x 1
    gradient = np.reshape(q-y, (len(state), 1))*state
    for i in range(len(state)):
        weights[:,action[i]] -= lr*gradient[i]

def getBin(s, intervals):
    for i in range(len(intervals)):
        if intervals[i].left <= s <= intervals[i].right:
            return i
    return 0

k = 5
for i in range(k):
    update(states_onehot, actions, rewards, nextstates_onehot, weights, 0.00001)

env = gym.make("CartPole-v1")
rewards = []
for episode in range(100):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        if (do_sigmoid):
            state = np.vectorize(sigmoid)(state)
        b1 = getBin(state[0], s1_cat.categories)
        b2 = getBin(state[1], s2_cat.categories)
        b3 = getBin(state[2], s3_cat.categories)
        b4 = getBin(state[3], s4_cat.categories)
        state = one_hot(b1,b2,b3,b4)

        action = np.argmax(state.dot(weights))
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        total_reward += reward

    rewards.append(total_reward)

print(np.mean(rewards))
print(np.std(rewards))