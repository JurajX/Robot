import itertools

import pytest
import robot.helpers as hlp
import robot.robot as robot
import torch
import yaml

cfg_name = 'test_config.yml'
proj_name = 'tests'
_, config_path = hlp.findProjectAndFilePaths(proj_name, [cfg_name])
with open(config_path[cfg_name], "r") as ymlfile:
    tmp = yaml.safe_load(ymlfile)
    cfg = tmp['test_set_get']
    panda = tmp['panda']

for key, value in cfg.items():
    cfg[key] = eval(value)


@pytest.fixture(scope='function')
def fct_robot():
    rbt = robot.Robot(panda['nLinks'], panda['directionOfGravity'], panda['rotationAxesOfJoints'], panda['frameCoordinates'], panda['dtype'])
    return rbt


class Test_RobotSet():

    @pytest.mark.parametrize('args',
                             itertools.product(cfg['directions'], cfg['dtypes'], cfg['r_grads']),
                             ids=lambda args: f"{type(args[0]).__name__}-f{str(args[1])[-2:]}-r_grad:{'T' if args[2] else 'F'}")
    def test_setGravAccel(self, fct_robot, monkeypatch, args):
        """The setGravAccel member function should set the gravAccel attribute of the Robot class."""
        direction = args[0]
        dtype = args[1]
        requires_grad = args[2]
        monkeypatch.setattr(fct_robot, 'dtype', dtype, raising=True)

        if type(direction) is torch.Tensor:
            expected = direction.to(dtype=dtype)
        else:
            expected = torch.tensor(direction, dtype=dtype)
        expected *= (fct_robot.G / expected.norm())

        fct_robot.setGravAccel(direction, requires_grad=requires_grad)
        assert type(fct_robot.gravAccel) is torch.nn.parameter.Parameter
        assert fct_robot.gravAccel.requires_grad is requires_grad
        assert fct_robot.gravAccel.allclose(expected)

    @pytest.mark.parametrize('args',
                             itertools.product(zip(cfg['n_links'], cfg['tensor1D']), cfg['dtypes'], cfg['r_grads']),
                             ids=lambda args: f"n:{args[0][0]}-f{str(args[1])[-2:]}-r_grad:{'T' if args[2] else 'F'}")
    def test_setMass(self, fct_robot, monkeypatch, args):
        """The setMass member function should set the mass attribute."""
        nLinks = args[0][0]
        mass = args[0][1]
        dtype = args[1]
        requires_grad = args[2]
        monkeypatch.setattr(fct_robot, 'nLinks', nLinks, raising=True)
        monkeypatch.setattr(fct_robot, 'dtype', dtype, raising=True)

        fct_robot.setMass(mass, requires_grad=requires_grad)

        assert type(fct_robot.mass) is torch.nn.parameter.Parameter
        assert fct_robot.mass.requires_grad is requires_grad
        assert fct_robot.mass.equal(mass.to(dtype=dtype))

    @pytest.mark.parametrize('args',
                             itertools.product(zip(cfg['n_links'], cfg['tensor3D']), cfg['dtypes'], cfg['r_grads']),
                             ids=lambda args: f"n:{args[0][0]}-f{str(args[1])[-2:]}-r_grad:{'T' if args[2] else 'F'}")
    def test_setCoM(self, fct_robot, monkeypatch, args):
        """The setInertialParams member function should set the linkCoM attribute."""
        nLinks = args[0][0]
        centreOfMass = args[0][1]
        dtype = args[1]
        requires_grad = args[2]
        monkeypatch.setattr(fct_robot, 'nLinks', nLinks, raising=True)
        monkeypatch.setattr(fct_robot, 'dtype', dtype, raising=True)

        fct_robot.setCoM(centreOfMass, requires_grad=requires_grad)

        assert type(fct_robot.linkCoM) is torch.nn.parameter.Parameter
        assert fct_robot.linkCoM.requires_grad is requires_grad
        assert fct_robot.linkCoM.equal(centreOfMass.to(dtype=dtype))

    @pytest.mark.parametrize('args',
                             itertools.product(zip(cfg['n_links'], cfg['tensor3D'], cfg['triangles']), cfg['dtypes'], cfg['r_grads'], [cfg['dim']]),
                             ids=lambda args: f"n:{args[0][0]}-f{str(args[1])[-2:]}-r_grad:{'T' if args[2] else 'F'}")
    def test_setPrincipalInertias(self, fct_robot, monkeypatch, args):
        """The setPrincipalInertias member function should set the J1J2 and the J1J2angle attributes of the Robot class."""
        DIM = args[3]
        nLinks = args[0][0]
        pInertias = args[0][1]
        triangle = args[0][2]
        dtype = args[1]
        requires_grad = args[2]
        monkeypatch.setattr(fct_robot, 'nLinks', nLinks, raising=True)
        monkeypatch.setattr(fct_robot, 'linksXdim', nLinks * DIM, raising=True)
        monkeypatch.setattr(fct_robot, 'dtype', dtype, raising=True)

        fct_robot.setPrincipalInertias(pInertias, requires_grad=requires_grad)

        assert type(fct_robot.triangle) is torch.nn.parameter.Parameter
        assert fct_robot.triangle.requires_grad is requires_grad
        assert fct_robot.triangle.equal(triangle.to(dtype=dtype))

    @pytest.mark.parametrize('args',
                             itertools.product(zip(cfg['n_links'], cfg['tensor3D']), cfg['dtypes'], cfg['r_grads'], [cfg['dim']]),
                             ids=lambda args: f"n:{args[0][0]}-f{str(args[1])[-2:]}-r_grad:{'T' if args[2] else 'F'}")
    def test_setRotOfPrincipalAxes(self, fct_robot, monkeypatch, args):
        """The setPrincipalInertias member function should set the J1J2 and the J1J2angle attributes of the Robot class."""
        DIM = args[3]
        nLinks = args[0][0]
        rotOfPrincipalAxes = args[0][1]
        dtype = args[1]
        requires_grad = args[2]
        monkeypatch.setattr(fct_robot, 'nLinks', nLinks, raising=True)
        monkeypatch.setattr(fct_robot, 'linksXdim', nLinks * DIM, raising=True)
        monkeypatch.setattr(fct_robot, 'dtype', dtype, raising=True)

        fct_robot.setRotOfPrincipalAxes(rotOfPrincipalAxes, requires_grad=requires_grad)

        assert type(fct_robot.rotationOfPrincipalAxes) is torch.nn.parameter.Parameter
        assert fct_robot.rotationOfPrincipalAxes.requires_grad is requires_grad
        assert fct_robot.rotationOfPrincipalAxes.equal(rotOfPrincipalAxes.to(dtype=dtype))

    @pytest.mark.parametrize('args',
                             itertools.product(zip(cfg['n_links'], cfg['tensor1D']), cfg['dtypes'], cfg['r_grads']),
                             ids=lambda args: f"n:{args[0][0]}-f{str(args[1])[-2:]}-r_grad:{'T' if args[2] else 'F'}")
    def test_setDamping(self, fct_robot, monkeypatch, args):
        """The setInertialParams member function should set the damping attribute."""
        nLinks = args[0][0]
        damping = args[0][1]
        dtype = args[1]
        requires_grad = args[2]
        monkeypatch.setattr(fct_robot, 'nLinks', nLinks, raising=True)
        monkeypatch.setattr(fct_robot, 'dtype', dtype, raising=True)

        fct_robot.setDamping(damping, requires_grad=requires_grad)

        assert type(fct_robot.damping) is torch.nn.parameter.Parameter
        assert fct_robot.damping.requires_grad is requires_grad
        assert fct_robot.damping.equal(damping.to(dtype=dtype))

    @pytest.mark.parametrize('args',
                             itertools.product(zip(cfg['n_links'], cfg['tensor1D'], cfg['tensor3D'], cfg['triangles']), cfg['dtypes'], cfg['r_grads'],
                                               [cfg['dim']]),
                             ids=lambda args: f"n:{args[0][0]}-f{str(args[1])[-2:]}-r_grad:{'T' if args[2] else 'F'}")
    def test_setInertialParams(self, fct_robot, monkeypatch, args):
        """The setInertialParams member function should set the mass, J1J2, J1J2angle, linkCoM, rotationOfPrincipalAxes, and damping attributes."""
        DIM = args[3]
        nLinks = args[0][0]
        mass = args[0][1]
        pInertias = args[0][2]
        centreOfMass = args[0][2]
        rotOfPrincipalAxes = args[0][2]
        triangle = args[0][3]
        damping = args[0][1]
        dtype = args[1]
        requires_grad = args[2]
        monkeypatch.setattr(fct_robot, 'nLinks', nLinks, raising=True)
        monkeypatch.setattr(fct_robot, 'linksXdim', nLinks * DIM, raising=True)
        monkeypatch.setattr(fct_robot, 'dtype', dtype, raising=True)

        fct_robot.setInertialParams(mass, pInertias, centreOfMass, rotOfPrincipalAxes, damping, requires_grad=requires_grad)

        assert type(fct_robot.mass) is torch.nn.parameter.Parameter
        assert type(fct_robot.triangle) is torch.nn.parameter.Parameter
        assert type(fct_robot.linkCoM) is torch.nn.parameter.Parameter
        assert type(fct_robot.rotationOfPrincipalAxes) is torch.nn.parameter.Parameter
        assert type(fct_robot.damping) is torch.nn.parameter.Parameter

        assert fct_robot.mass.requires_grad is requires_grad
        assert fct_robot.triangle.requires_grad is requires_grad
        assert fct_robot.linkCoM.requires_grad is requires_grad
        assert fct_robot.rotationOfPrincipalAxes.requires_grad is requires_grad
        assert fct_robot.damping.requires_grad is requires_grad

        assert fct_robot.mass.equal(mass.to(dtype=dtype))
        assert fct_robot.triangle.equal(triangle.to(dtype=dtype))
        assert fct_robot.linkCoM.equal(centreOfMass.to(dtype=dtype))
        assert fct_robot.rotationOfPrincipalAxes.equal(rotOfPrincipalAxes.to(dtype=dtype))
        assert fct_robot.damping.equal(damping.to(dtype=dtype))


class Test_RobotGet():

    @pytest.mark.parametrize('device', cfg['devices'], ids=lambda device: f"{device}")
    def test_device(self, fct_robot, device):
        """The device property should return the torch.device of the parameters of the class."""
        fct_robot = fct_robot.to(device=device)
        returned = fct_robot.device.type
        expected = torch.device(device).type
        assert returned == expected

    @pytest.mark.parametrize('args',
                             itertools.product(zip(cfg['n_links'], cfg['tensor3D']), cfg['dtypes'], cfg['r_grads'], [cfg['dim']]),
                             ids=lambda args: f"n:{args[0][0]}-f{str(args[1])[-2:]}-r_grad:{'T' if args[2] else 'F'}")
    def test_principalAxesInertia(self, fct_robot, monkeypatch, args):
        """The principalAxesInertia property should return the correct principal inertia of each link."""
        DIM = args[3]
        dtype = args[1]
        nLinks = args[0][0]
        pInertias = args[0][1]
        requires_grad = args[2]
        monkeypatch.setattr(fct_robot, 'nLinks', nLinks, raising=True)
        monkeypatch.setattr(fct_robot, 'linksXdim', nLinks * DIM, raising=True)
        monkeypatch.setattr(fct_robot, 'dtype', dtype, raising=True)

        fct_robot.setPrincipalInertias(pInertias, requires_grad=requires_grad)

        assert fct_robot.principalAxesInertia.allclose(pInertias.to(dtype=dtype))

    @pytest.mark.parametrize('args',
                             itertools.product(zip(cfg['n_links'], cfg['tensor3D'], cfg['rot_of_p_axes'], cfg['rot_mat']), cfg['dtypes'],
                                               [cfg['dim']]),
                             ids=lambda args: f"n:{args[0][0]}-f{str(args[1])[-2:]}")
    def test_inertia(self, fct_robot, monkeypatch, args):
        """The inertia property should return the correct inertia tensor for each link."""
        DIM = args[2]
        dtype = args[1]
        nLinks = args[0][0]
        pInertias = args[0][1]
        rotOfPrincipalAxes = args[0][2]
        rotMat = args[0][3]
        monkeypatch.setattr(fct_robot, 'nLinks', nLinks, raising=True)
        monkeypatch.setattr(fct_robot, 'linksXdim', nLinks * DIM, raising=True)
        monkeypatch.setattr(fct_robot, 'dtype', dtype, raising=True)

        fct_robot.setPrincipalInertias(pInertias)
        fct_robot.setRotOfPrincipalAxes(rotOfPrincipalAxes)
        fct_robot.SO3GEN.data = fct_robot.SO3GEN.to(dtype=dtype)
        expected = torch.einsum('nij, nj, nkj -> nik', rotMat, pInertias, rotMat)
        returned = fct_robot.inertia
        assert returned.allclose(expected.to(dtype=dtype), atol=1e-5)
