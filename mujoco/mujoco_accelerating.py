import os
import sys

sys.path.insert(0, os.path.dirname(sys.path[0]))

import time

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
    duration = 2

    q_min = torch.tensor(panda['q_min'], dtype=dtype)
    q_max = torch.tensor(panda['q_max'], dtype=dtype)
    span = (q_max - q_min)
    acceleration = 5 * span / duration

    for i in range(n_links):
        theta = torch.tensor([[0.] * n_links], dtype=dtype)
        theta[:, i] = q_min[i] + span[i] / 3
        dtheta = torch.tensor([[0.] * n_links], dtype=dtype)
        ddtheta = torch.tensor([[0.] * n_links], dtype=dtype)
        ddtheta[:, i] = acceleration[i]
        sim.data.qpos[:] = theta.squeeze(0).numpy()
        sim.data.qvel[:] = dtheta.squeeze(0).numpy()
        sim.data.qacc[:] = ddtheta.squeeze(0).numpy()
        t_start = time.time()
        while (time.time() - t_start < duration):
            theta = torch.from_numpy(sim.data.qpos[:]).unsqueeze(0)
            dtheta = torch.from_numpy(sim.data.qvel[:]).unsqueeze(0)
            sim.data.ctrl[:7] = rbt.getMotorTorque(theta, dtheta, ddtheta)[0].numpy()
            sim.step()
            viewer.render()


if __name__ == "__main__":
    main()
