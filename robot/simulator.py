import argparse

import torch
import yaml

import helpers as hlp
import robot as robot


def main(init):
    cfg_name = 'config.yml'
    proj_name = 'Robot'
    _, config_path = hlp.findProjectAndFilePaths(proj_name, [cfg_name])
    with open(config_path[cfg_name], "r") as ymlfile:
        panda = yaml.safe_load(ymlfile)['panda']

    setSize = 2**15
    batchSize = 2**5

    sim = robot.Robot(panda['nLinks'], panda['directionOfGravity'], panda['rotationAxesOfJoints'], panda['frameCoordinates'], panda['dtype'])
    sim.setInertialParams(panda['mass'], panda['linkCoM'], panda['principalInertias'], panda['rotationOfPrincipalAxes'], panda['damping'])
    loader = hlp.generateData(sim, panda, setSize, batchSize)

    rbt = robot.Robot(panda['nLinks'], panda['directionOfGravity'], panda['rotationAxesOfJoints'], panda['frameCoordinates'], panda['dtype'])
    rbt.setMass(panda['mass'], False)
    if init == 'warm':
        rbt.setDamping((1 - torch.randn([])) * torch.tensor(panda['damping']), True)
        rbt.setCoM((1 - torch.randn([]) * 1e-1) * torch.tensor(panda['linkCoM']), True)
        rbt.setPrincipalInertias((1 - torch.randn([]) * 1e-1) * torch.tensor(panda['principalInertias']), True)
        rbt.setRotOfPrincipalAxes((1 - torch.randn([]) * 1e-1) * torch.tensor(panda['rotationOfPrincipalAxes']), True)
        betas = [(0.9, 0.999), (0.89, 0.995), (0.88, 0.98), (0.87, 0.98)]
        lrs = [1e-3, 5e-4, 5e-5, 5e-6]
    else:
        betas = [(0.9, 0.999), (0.89, 0.995), (0.88, 0.99), (0.87, 0.98)]
        lrs = [5e-3, 5e-4, 5e-5, 5e-6]

    losses = []
    averages = [2e-4, 2e-6, 2e-8, 1e-9]

    for average, beta, lr in zip(averages, betas, lrs):
        avg_loss = 1.
        print(f"----- target average loss: {average:.0e}")
        optimiser = torch.optim.Adam(rbt.parameters(), lr=lr, betas=beta, eps=1e-08, weight_decay=0, amsgrad=True)
        while (avg_loss > average):
            avg_loss = hlp.learningLoop(rbt, optimiser, loader, losses)

    print(f"damping error:\n{rbt.damping.abs() - sim.damping}")
    print(f"learned CoM:\n{rbt.linkCoM}\ndesired CoM\n{sim.linkCoM}")
    print(f"learned inertia:\n{rbt.inertia}\ndesired inertia\n{sim.inertia}")
    return losses


if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Lear the inertial parameters of the panda robot in a simulator')
    p.add_argument("-i",
                   "--init",
                   type=str,
                   choices=['rand', 'warm'],
                   default='rand',
                   help="the initialisation of the inertial parameters, default='rand'")
    args = p.parse_args()

    losses = main(args.init)
    hlp.plotLoss(losses)
