import torch
import yaml

import src.robot as robot
import src.utils.find_paths as paths
import src.utils.random_vectors as vecs

cfg_name = 'test_config.yml'
proj_name = 'Robot'
_, config_path = paths.findProjectAndFilePaths(proj_name, [cfg_name])
with open(config_path[cfg_name], "r") as ymlfile:
    tmp = yaml.safe_load(ymlfile)
    cfg_s = tmp['single_pendulum']
    cfg_d = tmp['double_pendulum']

for key, value in cfg_s.items():
    cfg_s[key] = eval(value)

for key, value in cfg_d.items():
    cfg_d[key] = eval(value)


class Test_SinglePendulum():

    def test_mathematical(self):
        """The Robot class should correctly compute Lagrangian for the a mathematical pendulum."""
        sim = robot.Robot(cfg_s['n_links'], cfg_s['direction'], cfg_s['rot_axes'], cfg_s['f_coos'], cfg_s['dtype'])
        sim.setInertialParams(cfg_s['mass'], cfg_s['com'], cfg_s['p_inertia'], cfg_s['rot_of_p_axes'], cfg_s['damping'])
        sim.triangle.data = torch.zeros_like(sim.triangle.data)    # Mathematical pendulum => no inertia tensor

        thetas = vecs.generateRandomVectors(cfg_s['set_size'], [-cfg_s['pi']], [cfg_s['pi']], cfg_s['dtype'])
        dthetas = vecs.generateRandomVectors(cfg_s['set_size'], [-cfg_s['max_vel']], [cfg_s['max_vel']], cfg_s['dtype'])

        T = 0.5 * cfg_s['mass'][0] * (cfg_s['com'].norm() * dthetas.squeeze()).pow(2)
        V = cfg_s['mass'][0] * cfg_s['g'] * cfg_s['com'].norm() * (-1) * torch.cos(thetas.squeeze())
        expected = T - V
        returned = sim.getLagrangian(thetas, dthetas)
        assert expected.allclose(returned)

    def test_physical_com(self):
        """The Robot class should correctly compute Lagrangian for the a physical pendulum rotating around it's CoM."""
        sim = robot.Robot(cfg_s['n_links'], cfg_s['direction'], cfg_s['rot_axes'], cfg_s['f_coos'], cfg_s['dtype'])
        sim.setInertialParams(cfg_s['mass'], cfg_s['com'], cfg_s['p_inertia'], cfg_s['rot_of_p_axes'], cfg_s['damping'])
        sim.mass.data = torch.zeros_like(sim.mass.data)    # rotation around CoM => no potential energy

        thetas = vecs.generateRandomVectors(cfg_s['set_size'], [-cfg_s['pi']], [cfg_s['pi']], cfg_s['dtype'])
        dthetas = vecs.generateRandomVectors(cfg_s['set_size'], [-cfg_s['max_vel']], [cfg_s['max_vel']], cfg_s['dtype'])

        projectedInertia = torch.einsum('ni, nij, nj -> ', cfg_s['rot_axes'], sim.inertia, cfg_s['rot_axes'])
        T = 0.5 * projectedInertia * dthetas.squeeze().pow(2)
        V = torch.zeros_like(T)    # rotation around CoM => no potential energy
        expected = T - V
        returned = sim.getLagrangian(thetas, dthetas)
        assert expected.allclose(returned)

    def test_physical(self):
        """The Robot class should correctly compute Lagrangian for the a physical pendulum."""
        sim = robot.Robot(cfg_s['n_links'], cfg_s['direction'], cfg_s['rot_axes'], cfg_s['f_coos'], cfg_s['dtype'])
        sim.setInertialParams(cfg_s['mass'], cfg_s['com'], cfg_s['p_inertia'], cfg_s['rot_of_p_axes'], cfg_s['damping'])

        thetas = vecs.generateRandomVectors(cfg_s['set_size'], [-cfg_s['pi']], [cfg_s['pi']], cfg_s['dtype'])
        dthetas = vecs.generateRandomVectors(cfg_s['set_size'], [-cfg_s['max_vel']], [cfg_s['max_vel']], cfg_s['dtype'])

        projectedInertia = torch.einsum('ni, nij, nj -> ', cfg_s['rot_axes'], sim.inertia, cfg_s['rot_axes'])
        T = 0.5 * cfg_s['mass'][0] * (cfg_s['com'].norm() * dthetas.squeeze()).pow(2) + 0.5 * projectedInertia * dthetas.squeeze().pow(2)
        V = cfg_s['mass'][0] * cfg_s['g'] * cfg_s['com'].norm() * (-1) * torch.cos(thetas.squeeze())
        expected = T - V
        returned = sim.getLagrangian(thetas, dthetas)
        assert expected.allclose(returned)


class Test_DoublePendulum():

    def test_mathematical(self):
        """The Robot class should correctly compute Lagrangian for the a double mathematical pendulum."""
        sim = robot.Robot(cfg_d['n_links'], cfg_d['direction'], cfg_d['rot_axes_planar'], cfg_d['f_coos_m'], cfg_d['dtype'])
        sim.setInertialParams(cfg_d['mass'], cfg_d['com'], cfg_d['p_inertias_math'], cfg_d['rot_of_p_axes'], cfg_d['damping'])
        sim.triangle.data = torch.zeros_like(sim.triangle.data)

        thetas = vecs.generateRandomVectors(cfg_d['set_size'], [-cfg_d['pi']] * 2, [cfg_d['pi']] * 2, cfg_d['dtype'])
        dthetas = vecs.generateRandomVectors(cfg_d['set_size'], [-cfg_d['max_vel']] * 2, [cfg_d['max_vel']] * 2, cfg_d['dtype'])

        V1 = cfg_d['mass'][0] * cfg_d['g'] * cfg_d['com'][0].norm() * (-torch.cos(thetas[:, 0]))
        V2 = cfg_d['mass'][1] * cfg_d['g'] * cfg_d['com'][0].norm() * (-torch.cos(thetas[:, 0]))
        V2 += cfg_d['mass'][1] * cfg_d['g'] * cfg_d['com'][1].norm() * (-torch.cos(thetas[:, 0] + thetas[:, 1]))

        tmp1 = cfg_d['mass'][0] * cfg_d['com'][0].norm().pow(2)
        tmp1 = tmp1 + cfg_d['mass'][1] * (cfg_d['com'][0].norm().pow(2) + cfg_d['com'][1].norm().pow(2) +
                                          2 * cfg_d['com'][0].norm() * cfg_d['com'][1].norm() * torch.cos(thetas[:, 1]))
        tmp2 = cfg_d['mass'][1] * cfg_d['com'][1].norm().pow(2)
        tmp3 = tmp2 + cfg_d['mass'][1] * cfg_d['com'][0].norm() * cfg_d['com'][1].norm() * torch.cos(thetas[:, 1])
        T = 0.5 * (tmp1 * dthetas[:, 0].pow(2) + tmp2 * dthetas[:, 1].pow(2) + 2 * tmp3 * dthetas[:, 0] * dthetas[:, 1])

        expected = T - V1 - V2
        returned = sim.getLagrangian(thetas, dthetas)
        assert expected.allclose(returned)

    def test_physical_planar(self):
        """The Robot class should correctly compute Lagrangian for the a double physical pendulum restricted to a plane."""
        sim = robot.Robot(cfg_d['n_links'], cfg_d['direction'], cfg_d['rot_axes_planar'], cfg_d['f_coos_p'], cfg_d['dtype'])
        sim.setInertialParams(cfg_d['mass'], cfg_d['com'], cfg_d['p_inertias_phys'], cfg_d['rot_of_p_axes'], cfg_d['damping'])

        thetas = vecs.generateRandomVectors(cfg_d['set_size'], [-cfg_d['pi']] * 2, [cfg_d['pi']] * 2, cfg_d['dtype'])
        dthetas = vecs.generateRandomVectors(cfg_d['set_size'], [-cfg_d['max_vel']] * 2, [cfg_d['max_vel']] * 2, cfg_d['dtype'])

        V1 = cfg_d['mass'][0] * cfg_d['g'] * cfg_d['com'][0].norm() * (-torch.cos(thetas[:, 0]))
        V2 = cfg_d['mass'][1] * cfg_d['g'] * cfg_d['f_coos_p'][0 + 1].norm() * (-torch.cos(thetas[:, 0]))
        V2 += cfg_d['mass'][1] * cfg_d['g'] * cfg_d['com'][1].norm() * (-torch.cos(thetas[:, 0] + thetas[:, 1]))

        projectedInertia = torch.einsum('ni, nij, nj -> n', cfg_d['rot_axes_planar'], sim.inertia, cfg_d['rot_axes_planar'])
        T1 = 0.5 * projectedInertia[0] * dthetas[:, 0].pow(2) + 0.5 * projectedInertia[1] * dthetas.sum(dim=1).pow(2)
        tmp11 = cfg_d['f_coos_p'][0 + 1].norm().pow(2) + cfg_d['com'][1].norm().pow(2)
        tmp12 = 2 * cfg_d['f_coos_p'][0 + 1].norm() * cfg_d['com'][1].norm() * torch.cos(thetas[:, 1])
        tmp1 = cfg_d['mass'][1] * (tmp11+tmp12)
        tmp1 = tmp1 + cfg_d['mass'][0] * cfg_d['com'][0].norm().pow(2)
        tmp2 = cfg_d['mass'][1] * cfg_d['com'][1].norm().pow(2)
        tmp3 = tmp2 + cfg_d['mass'][1] * cfg_d['f_coos_p'][0 + 1].norm() * cfg_d['com'][1].norm() * torch.cos(thetas[:, 1])
        T2 = 0.5 * (tmp1 * dthetas[:, 0].pow(2) + tmp2 * dthetas[:, 1].pow(2) + 2 * tmp3 * dthetas[:, 0] * dthetas[:, 1])

        expected = T1 + T2 - V1 - V2
        returned = sim.getLagrangian(thetas, dthetas)
        assert expected.allclose(returned)

    def test_physical_non_planar(self):
        """The Robot class should correctly compute Lagrangian for the a double physical pendulum not restricted to a plane."""
        sim = robot.Robot(cfg_d['n_links'], cfg_d['direction'], cfg_d['rot_axes_non_planar'], cfg_d['f_coos_p'], cfg_d['dtype'])
        sim.setInertialParams(cfg_d['mass'], cfg_d['com'], cfg_d['p_inertias_phys'], cfg_d['rot_of_p_axes'], cfg_d['damping'])

        thetas = vecs.generateRandomVectors(cfg_d['set_size'], [-cfg_d['pi']] * 2, [cfg_d['pi']] * 2, cfg_d['dtype'])
        dthetas = vecs.generateRandomVectors(cfg_d['set_size'], [-cfg_d['max_vel']] * 2, [cfg_d['max_vel']] * 2, cfg_d['dtype'])

        V1 = cfg_d['mass'][0] * cfg_d['g'] * cfg_d['com'][0].norm() * (-torch.cos(thetas[:, 0]))
        V2 = cfg_d['mass'][1] * cfg_d['g'] * cfg_d['f_coos_p'][0 + 1].norm() * (-torch.cos(thetas[:, 0]))
        V3 = cfg_d['mass'][1] * cfg_d['g'] * cfg_d['com'][1].norm() * (-torch.cos(thetas[:, 0]) * torch.cos(thetas[:, 1]))

        Lq = torch.einsum('ijk, ni -> njk', sim.SO3GEN, cfg_d['rot_axes_non_planar']).unsqueeze(0) * thetas.unsqueeze(-1).unsqueeze(-1)
        Lq = Lq.reshape([cfg_d['set_size'] * cfg_d['n_links'], 3, 3])
        Lq = Lq.reshape([cfg_d['set_size'], cfg_d['n_links'], 3, 3])
        R = torch.matrix_exp(Lq)
        R12 = R[:, 0].matmul(R[:, 1])
        Inertia = torch.empty(100, 2, 2, dtype=cfg_d['dtype'])
        tmp = sim.inertia[0] + R12.matmul(sim.inertia[1]).matmul(R12.transpose(1, 2))
        Inertia[:, 0, 0] = tmp.matmul(cfg_d['rot_axes_non_planar'][0]).matmul(cfg_d['rot_axes_non_planar'][0])
        tmp = R[:, 1].matmul(sim.inertia[1]).matmul(R12.transpose(1, 2))
        Inertia[:, 1, 0] = tmp.matmul(cfg_d['rot_axes_non_planar'][0]).matmul(cfg_d['rot_axes_non_planar'][1])
        tmp = R12.matmul(sim.inertia[1]).matmul(R[:, 1].transpose(1, 2))
        Inertia[:, 0, 1] = tmp.matmul(cfg_d['rot_axes_non_planar'][1]).matmul(cfg_d['rot_axes_non_planar'][0])
        Inertia[:, 1, 1] = sim.inertia[1].matmul(cfg_d['rot_axes_non_planar'][1]).matmul(cfg_d['rot_axes_non_planar'][1])
        T1 = 0.5 * torch.einsum('bn, bnm, bm -> b', dthetas, Inertia, dthetas)
        tmp1 = cfg_d['mass'][0] * cfg_d['com'][0].norm().pow(2)
        tmp1 = tmp1 + cfg_d['mass'][1] * (cfg_d['f_coos_p'][0 + 1].norm() + cfg_d['com'][1].norm() * torch.cos(thetas[:, 1])).pow(2)
        tmp2 = cfg_d['mass'][1] * cfg_d['com'][1].norm().pow(2)
        T2 = 0.5 * (tmp1 * dthetas[:, 0].pow(2) + tmp2 * dthetas[:, 1].pow(2))

        expected = T1 + T2 - V1 - V2 - V3
        returned = sim.getLagrangian(thetas, dthetas)
        assert expected.allclose(returned)
