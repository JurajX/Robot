import time

import matplotlib.pyplot as plt
import torch

from . import random_vectors as rv


def printRunningLoss(running_loss, num_tenths, epoch=None):
    """Print the running loss every on the same line.
    Arguments:
        epoch        - epoch of the training; int
        running_loss - the loss to be printed; list
        num_tenths   - progress of the training in fraction of the whole; int in the interval [0, 10]
    """
    if epoch is not None:
        print(f"\repoch: {epoch}; average loss: {running_loss:.3e}, progress: [{'='*num_tenths + '-'*(10-num_tenths)}]", end='')
    else:
        print(f"\raverage loss: {running_loss:.3e}, progress: [{'='*num_tenths + '-'*(10-num_tenths)}]", end='')


def plotLoss(loss, offset=0):
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Training Loss')
    ax.plot(range(offset, len(loss)), loss[offset:])
    ax.set_xlabel('# of batches seen')
    ax.set_ylabel('avg. loss per batch')
    plt.yscale('log')
    plt.show()


def generateData(sim, cfg, set_size, chunk_size=2**15):
    """Generate random data from a simulator for learning dynamic parameters.
    Arguments:
        sim        - simulator used to generate the data
        cfg        - configuration file for a robot
        set_size   - size of the dataset
        chunk_size - the data is generated in chunks of size chunk_size (default=2**15)
    Returns:
        tuple      - containing the generated data for gravity_direction, thetas, dthetas, ddthetas, appliedTorques
    """
    if set_size < chunk_size:
        chunk_size = set_size
    gravity_direction = rv.generateUnitVectors([set_size, 3], sim.dtype)
    thetas = rv.generateRandomVectors(set_size, cfg['q_min'], cfg['q_max'], sim.dtype)
    dthetas = rv.generateRandomVectors(set_size, cfg['dq_min'], cfg['dq_max'], sim.dtype)
    ddthetas = rv.generateRandomVectors(set_size, cfg['ddq_min'], cfg['ddq_max'], sim.dtype)
    appliedTorques = torch.empty(thetas.shape, dtype=sim.dtype)

    with torch.no_grad():
        for b in range(0, set_size, chunk_size):
            e = b + chunk_size
            appliedTorques[b:e] = sim.getMotorTorque(thetas[b:e], dthetas[b:e], ddthetas[b:e], gravity_direction[b:e])
    return gravity_direction, thetas, dthetas, ddthetas, appliedTorques


def generateDataLoader(sim, cfg, set_size, batch_size, chunk_size=2**15):
    """Generate random data from a simulator for learning dynamic parameters and store them in a data loader.
    Arguments:
        sim        - simulator used to generate the data
        cfg        - configuration file for a robot
        set_size   - size of the dataset
        batch_size - batch size argument for data loader
        chunk_size - the data is generated in chunks of size chunk_size (default=2**15)
    Returns:
        torch.utils.data.DataLoader - containing the generated data
    """
    gravity_direction, thetas, dthetas, ddthetas, appliedTorques = generateData(sim, cfg, set_size, chunk_size)
    dataset = torch.utils.data.TensorDataset(thetas, dthetas, ddthetas, appliedTorques, gravity_direction)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    return loader


def generateDataKin(sim, cfg, set_size, chunk_size=2**15):
    """Generate random data from a simulator for learning kinematic parameters.
    Arguments:
        sim        - simulator used to generate the data
        cfg        - configuration file for a robot
        set_size   - size of the dataset
        chunk_size - the data is generated in chunks of size chunk_size (default=2**15)
    Returns:
        tuple      - containing the generated data for thetas, coordinatesOfEE
    """
    if set_size < chunk_size:
        chunk_size = set_size
    thetas = rv.generateRandomVectors(set_size, cfg['q_min'], cfg['q_max'], sim.dtype)
    coordinatesOfEE = torch.empty((set_size, cfg['dim']), dtype=sim.dtype)

    with torch.no_grad():
        for b in range(0, set_size, chunk_size):
            e = b + chunk_size
            coordinatesOfEE[b:e] = sim.getCoordinatesOfEE(thetas[b:e])
    return thetas, coordinatesOfEE


def generateDataLoaderKin(sim, cfg, set_size, batch_size, chunk_size=2**15):
    """Generate random data from a simulator for learning kinematic parameters and store them in a data loader.
    Arguments:
        sim        - simulator used to generate the data
        cfg        - configuration file for a robot
        set_size   - size of the dataset
        batch_size - batch size argument for data loader
        chunk_size - the data is generated in chunks of size chunk_size (default=2**15)
    Returns:
        torch.utils.data.DataLoader - containing the generated data
    """
    thetas, coordinatesOfEE = generateDataKin(sim, cfg, set_size, chunk_size)
    dataset = torch.utils.data.TensorDataset(thetas, coordinatesOfEE)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    return loader


def learningLoop(agent, optimiser, loader, losses):
    """Learning loop iterating over the whole dataset onece for the agent.
    Arguments:
        agent     - the learning agent
        optimiser - optimiser used in learning; assumes that it contains the agent parameters
        loader    - data loader containing the dataset
        losses    - list where average batch loses are appended
    Returns:
        avg_loss  - average loss of the whole dataset
    """
    tenthOfData = len(loader.dataset) // (loader.batch_size * 10)
    start = time.time()
    criterion = torch.nn.MSELoss()
    avg_loss = 0.0
    for i, data in enumerate(loader, 1):
        theta = data[0].to(device=agent.device)
        dtheta = data[1].to(device=agent.device)
        ddtheta = data[2].to(device=agent.device)
        appliedTorque = data[3].to(device=agent.device)
        gravityDirection = data[4].to(device=agent.device)

        optimiser.zero_grad()
        appliedTorque_est = agent.getMotorTorque(theta, dtheta, ddtheta, directionOfGravity=gravityDirection)
        loss = criterion(appliedTorque_est, appliedTorque)
        loss.backward()
        optimiser.step()

        with torch.no_grad():
            avg_loss = avg_loss + (loss.item() - avg_loss) / i
            losses.append(loss.item())

        if (i % tenthOfData == 0):
            printRunningLoss(avg_loss, i // tenthOfData)
    print("")
    end = time.time()
    print(f"elapsed time {(end-start):.2f}s.")
    return avg_loss


class MSLELoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, actual):
        return self.mse(torch.log(pred + 1), torch.log(actual + 1))


def learningLoopKin(agent, optimiser, loader, losses):
    """Learning loop iterating over the whole dataset onece for the agent.
    Arguments:
        agent     - the learning agent
        optimiser - optimiser used in learning; assumes that it contains the agent parameters
        loader    - data loader containing the dataset
        losses    - list where average batch loses are appended
    Returns:
        avg_loss  - average loss of the whole dataset
    """
    tenthOfData = len(loader.dataset) // (loader.batch_size * 10)
    start = time.time()
    criterion = MSLELoss()
    avg_loss = 0.0
    for i, data in enumerate(loader, 1):
        theta = data[0].to(device=agent.device)
        coordinatesOfEE = data[1].to(device=agent.device)

        optimiser.zero_grad()
        coordinatesOfEE_est = agent.getCoordinatesOfEE(theta)
        loss = criterion(coordinatesOfEE_est, coordinatesOfEE)
        loss.backward()
        optimiser.step()

        with torch.no_grad():
            avg_loss = avg_loss + (loss.item() - avg_loss) / i
            losses.append(loss.item())

        if (i % tenthOfData == 0):
            printRunningLoss(avg_loss, i // tenthOfData)
    print("")
    end = time.time()
    print(f"elapsed time {(end-start):.2f}s.")
    return avg_loss
