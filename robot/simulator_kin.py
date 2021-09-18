import torch
import yaml

import helpers as hlp
import robot as robot


def main():
    cfg_name = 'config.yml'
    proj_name = 'Robot'
    _, config_path = hlp.findProjectAndFilePaths(proj_name, [cfg_name])
    with open(config_path[cfg_name], "r") as ymlfile:
        panda = yaml.safe_load(ymlfile)['panda']

    setSize = 2**15
    batchSize = 2**5

    sim = robot.Robot(panda['nLinks'], panda['directionOfGravity'], panda['rotationAxesOfJoints'], panda['frameCoordinates'], panda['dtype'])
    sim.setInertialParams(panda['mass'], panda['linkCoM'], panda['principalInertias'], panda['rotationOfPrincipalAxes'], panda['damping'])
    loader = hlp.generateDataKin(sim, panda, setSize, batchSize)

    rbt = robot.Robot(panda['nLinks'], panda['directionOfGravity'], panda['rotationAxesOfJoints'], panda['frameCoordinates'], panda['dtype'])
    rbt.setInertialParams(panda['mass'], panda['linkCoM'], panda['principalInertias'], panda['rotationOfPrincipalAxes'], panda['damping'])
    std = 0.001
    rbt.setRotationAxesOfJoints(torch.randn(rbt.rotationAxesOfJoints.shape) * std + torch.tensor(panda['rotationAxesOfJoints']), True)
    rbt.setFrameCoordinates(torch.randn(rbt.frameCoordinates.shape) * std + torch.tensor(panda['frameCoordinates']), True)

    betas = [(0.9, 0.999), (0.89, 0.995), (0.88, 0.98), (0.87, 0.98)]
    lrs = [1e-5, 1e-6, 5e-8, 1e-10]

    losses = []
    averages = [2e-4, 2e-6, 2e-7, 2e-8]

    for average, beta, lr in zip(averages, betas, lrs):
        avg_loss = 1.
        print(f"----- target average loss: {average:.0e}")
        optimiser = torch.optim.Adam(rbt.parameters(), lr=lr, betas=beta, eps=1e-08, weight_decay=0, amsgrad=True)
        while (avg_loss > average):
            avg_loss = hlp.learningLoopKin(rbt, optimiser, loader, losses)
    rbt.normaliseRotationAxesOfJoints()
    print(f"learned rotationAxesOfJoints:\n{rbt.rotationAxesOfJoints}\ndesired rotationAxesOfJoints\n{sim.rotationAxesOfJoints}")
    print(f"learned frameCoordinates:\n{rbt.frameCoordinates}\ndesired frameCoordinates\n{sim.frameCoordinates}")
    print(f"MSRE error for rotationAxesOfJoints is: {(rbt.rotationAxesOfJoints - sim.rotationAxesOfJoints).norm().sqrt()}")
    print(f"MSRE error for frameCoordinates is:     {(rbt.frameCoordinates - sim.frameCoordinates).norm().sqrt()}")
    return losses


if __name__ == "__main__":
    losses = main()
    hlp.plotLoss(losses)
