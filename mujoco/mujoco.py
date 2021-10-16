import os
import sys

sys.path.insert(0, os.path.dirname(sys.path[0]))

from math import pi

import mujoco_py
import torch
import yaml

import src.robot as robot
import src.utils.find_paths as paths


def main():
    proj_name = 'Robot'
    cfg_name = 'mujoco_config.yml'
    xml_name = 'panda.xml'
    dtype = torch.float64

    _, file_paths = paths.findProjectAndFilePaths(proj_name, [cfg_name, xml_name])
    with open(file_paths[cfg_name], "r") as ymlfile:
        panda = yaml.safe_load(ymlfile)['panda']

    xml_path = file_paths[xml_name]
    model = mujoco_py.load_model_from_path(xml_path)

    rbt = robot.Robot(panda['nLinks'], panda['directionOfGravity'], panda['rotationAxesOfJoints'], panda['frameCoordinates'], panda['dtype64'])
    rbt.setInertialParams(panda['mass'], panda['linkCoM'], panda['principalInertias'], panda['rotationOfPrincipalAxes'], panda['damping'])
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)

    n_links = sim.data.qpos.shape[0]
    n_steps = 10**4

    lin_space = torch.linspace(0, 2 * pi, n_steps, dtype=dtype)
    cos = lin_space.cos()
    sin = lin_space.sin()

    q_min = torch.tensor(panda['q_min'], dtype=dtype)
    q_max = torch.tensor(panda['q_max'], dtype=dtype)
    span = (q_max - q_min)
    middle = (q_max+q_min) / 2
    thetas = middle + cos.unsqueeze(1) * span.unsqueeze(0) / 2
    dthetas = -sin.unsqueeze(1) * span.unsqueeze(0) / 2
    ddthetas = -cos.unsqueeze(1) * span.unsqueeze(0) / 2

    sim.data.qpos[:] = q_max.numpy()
    sim.data.qvel[:] = (-sin[0] * span).numpy()
    sim.data.qacc[:] = ddthetas[0].numpy()
    for des_theta, des_dtheta, des_ddtheta in zip(thetas.unsqueeze(1), dthetas.unsqueeze(1), ddthetas.unsqueeze(1)):
        # theta = torch.from_numpy(sim.data.qpos[:])
        # dtheta = torch.from_numpy(sim.data.qvel[:])
        # torque = rbt.getMotorTorque(theta.unsqueeze(0), dtheta.unsqueeze(0), des_ddtheta)[0]
        # torque += 10 * (des_theta[0] - theta) + 10 * (des_dtheta[0] - dtheta)
        # sim.data.ctrl[:7] = torque.numpy()
        sim.data.qpos[:] = des_theta[0]
        sim.data.qvel[:] = torch.tensor([0.] * n_links, dtype=dtype).numpy()
        sim.data.qacc[:] = torch.tensor([0.] * n_links, dtype=dtype).numpy()
        sim.step()
        viewer.render()


if __name__ == "__main__":
    main()
