from rnn import RNN
import numpy as np
import itertools


np.set_printoptions(precision=2)


def run01test(nBatch, batchSize, stateDim, pattern, predictLength, rate=1):
    assert len(pattern) <= batchSize
    x = list(itertools.islice(itertools.cycle(pattern), batchSize - batchSize%len(pattern) + 1))
    x = np.array(x)

    rnn = RNN(inputDim=2, stateDim=stateDim, rate=rate)

    for i in range(nBatch):
        rnn.update(x)

    print(rnn.predict(start=pattern[0], nStep=predictLength))
    print(rnn.gradientNorm())
    print(rnn.dE_dU, rnn.dE_dV, rnn.dE_dW, sep='\n\n')


configurations = {}
configurations['01'] = dict(nBatch=200, batchSize=20, stateDim=5, pattern=[0,1], predictLength=10)
configurations['010'] = dict(nBatch=200, batchSize=20, stateDim=29, pattern=[0,1,0], predictLength=10)
configurations['0011'] = dict(nBatch=500, batchSize=20, stateDim=40, pattern=[0,0,1,1], predictLength=10)
configurations['0111'] = dict(nBatch=500, batchSize=20, stateDim=35, pattern=[0,1,1,1], predictLength=10)
configurations['1101 failed'] = dict(nBatch=10000, batchSize=20, stateDim=40, pattern=[1,1,0,1], predictLength=10)
configurations['1101'] = dict(nBatch=10000, batchSize=20, stateDim=50, pattern=[1,1,0,1], predictLength=10)
configurations['1101.a'] = dict(nBatch=2000, batchSize=20, stateDim=60, pattern=[1,1,0,1], predictLength=10)
configurations['1101.b failed'] = dict(nBatch=10000000, batchSize=20, stateDim=70, pattern=[1,1,0,1], predictLength=10)
configurations['1101.c'] = dict(nBatch=10000, batchSize=20, stateDim=50, pattern=[1,1,0,1], predictLength=10, rate=0.1)
configurations['1101.d'] = dict(nBatch=7000, batchSize=20, stateDim=70, pattern=[1,1,0,1], predictLength=10, rate=0.1)

def runCircTest():
    r0 = 10.0
    r1 = 10.0
    nTrainingPoint = 1000000
    batchSize = 30
    deltaTheta = 0.25
    theta = 0.0
    data = []

    rnn = RNN(inputDim=2, stateDim=20, rate=0.1)

    for i in range(0, nTrainingPoint):
        theta = (theta + deltaTheta) % (2*np.pi)
        x = r0 * np.cos(theta)
        y = r1 * np.sin(theta)
        data.append(np.array([x,y]))

    for i in range(0, nTrainingPoint, batchSize):
        batch = data[i:i+batchSize]
        rnn.update(batch)

    result = rnn.predict(start=np.array([r0, 0.0]), nStep=10)
    for x,y in result: print(x,y)

runCircTest()
