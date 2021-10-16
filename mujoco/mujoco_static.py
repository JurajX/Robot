import os
import sys

sys.path.insert(0, os.path.dirname(sys.path[0]))

import time

import mujoco_py
import torch
import yaml

import src.robot as robot
import src.utils.find_paths as paths
import src.utils.random_vectors as rvecs


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
    n_poses = 10
    duration = 2
    for _ in range(n_poses):
        theta = rvecs.generateRandomVectors(1, panda['q_min'], panda['q_max'], dtype)
        dtheta = torch.tensor([[0.] * n_links], dtype=dtype)
        ddtheta = torch.tensor([[0.] * n_links], dtype=dtype)
        sim.data.qpos[:] = theta.squeeze(0).numpy()
        sim.data.qvel[:] = dtheta.squeeze(0).numpy()
        sim.data.qacc[:] = ddtheta.squeeze(0).numpy()
        t_start = time.time()
        while (time.time() - t_start < duration):
            sim.data.ctrl[:7] = rbt.getMotorTorque(theta, dtheta, ddtheta)[0].numpy()
            sim.step()
            viewer.render()


if __name__ == "__main__":
    main()
