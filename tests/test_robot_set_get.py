import pytest
import torch
import robot.robot as robot
import itertools

# =========== TEST CONSTANTS ===========
DIM = 3
N_LINKS = [1, 3, 5, 7]
REQUIRES_GRAD = (True, False)

DTYPES = [torch.float32, torch.float64]
DEVICES = ['cpu', 'gpu'] if torch.cuda.is_available() else ['cpu']

DIRECTIONS = [torch.tensor([0., 0., 5.]), [-3., 0., 0.], (1., 1., 0.)]
TENSORS_1D = [torch.tensor(range(1, n + 1), dtype=torch.float32) for n in N_LINKS]
TENSORS_3D = [torch.tensor([[3., 4., 5.]]).expand(n, -1) for n in N_LINKS]

PI = 4 * torch.atan(torch.tensor(1.0, dtype=torch.float64))
ANGLES = [torch.tensor([PI / 2]).expand(n, -1) for n in N_LINKS]

tmp = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [-1., 0., 0.], [0., -1., 0.], [0., 0., -1.], [0., 1., 0.]])
ROT_AXES = [tmp[:n] for n in N_LINKS]
ROT_OF_P_AXES = [(PI/2) * tmp[:n] for n in N_LINKS]

tmp = torch.tensor([[[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], [[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]],
                    [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], [[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]],
                    [[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]], [[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]],
                    [[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]])
ROT_MAT_INERTIA = [tmp[:n] for n in N_LINKS]

tmp = [[0., 0., 0.1], [0., 0., 0.1], [0., 0., 0.1], [0., 0., 0.1], [0., 0., 0.1], [0., 0., 0.1], [0., 0., 0.1]]
FRAME_COORDINATES = [tmp[:n] for n in N_LINKS]

# =========== FIXTURES ===========
FIXTURE_N_LINKS = 7
FIXTURE_GRAVITY_DIRECTION = [0., 0., -1.]
FIXTURE_ROT_AXES_OF_JOINTS = [[0., 0., 1.], [0., 1., 0.], [0., 0., 1.], [0., -1., 0.], [0., 0., 1.], [0., -1., 0.], [0., 0., -1.]]
FIXTURE_FRAME_COORDINATES = [[0., 0., 0.1], [0., 0., 0.233], [0., 0., 0.1], [0.088, 0., 0.216], [-0.088, 0., 0.1], [0., 0., 0.284],
                             [0.088, 0., -0.107]]
FIXTURE_DTYPE = torch.float32


@pytest.fixture(scope='function')
def fct_robot():
    rbt = robot.Robot(FIXTURE_N_LINKS, FIXTURE_GRAVITY_DIRECTION, FIXTURE_ROT_AXES_OF_JOINTS, FIXTURE_FRAME_COORDINATES, FIXTURE_DTYPE)
    return rbt


# =========== TESTING ===========


class Test_RobotSetGet():

    @pytest.mark.parametrize('args',
                             itertools.product(DIRECTIONS, DTYPES, REQUIRES_GRAD),
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
                             itertools.product(zip(N_LINKS, TENSORS_3D, ANGLES), DTYPES, REQUIRES_GRAD, [DIM]),
                             ids=lambda args: f"n:{args[0][0]}-f{str(args[1])[-2:]}-r_grad:{'T' if args[2] else 'F'}")
    def test_setPrincipalInertias(self, fct_robot, monkeypatch, args):
        """The setPrincipalInertias member function should set the J1J2 and the J1J2angle attributes of the Robot class."""
        DIM = args[3]
        nLinks = args[0][0]
        pInertias = args[0][1]
        angles = args[0][2]
        dtype = args[1]
        requires_grad = args[2]
        monkeypatch.setattr(fct_robot, 'nLinks', nLinks, raising=True)
        monkeypatch.setattr(fct_robot, 'linksXdim', nLinks * DIM, raising=True)
        monkeypatch.setattr(fct_robot, 'dtype', dtype, raising=True)

        fct_robot.setPrincipalInertias(pInertias, requires_grad=requires_grad)

        assert type(fct_robot.J1J2) is torch.nn.parameter.Parameter
        assert fct_robot.J1J2.requires_grad is requires_grad
        assert fct_robot.J1J2.equal(pInertias[:, :-1].to(dtype=dtype))

        assert type(fct_robot.J1J2angle) is torch.nn.parameter.Parameter
        assert fct_robot.J1J2angle.requires_grad is requires_grad
        assert fct_robot.J1J2angle.allclose(angles.to(dtype=dtype))

    @pytest.mark.parametrize('args',
                             itertools.product(zip(N_LINKS, TENSORS_3D), DTYPES, REQUIRES_GRAD, [DIM]),
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
                             itertools.product(zip(N_LINKS, TENSORS_1D, TENSORS_3D), DTYPES, REQUIRES_GRAD, [DIM]),
                             ids=lambda args: f"n:{args[0][0]}-f{str(args[1])[-2:]}-r_grad:{'T' if args[2] else 'F'}")
    def test_setInertialParams(self, fct_robot, monkeypatch, args):
        """The setInertialParams member function should set the mass, J1J2, J1J2angle, linkCoM, rotationOfPrincipalAxes, and damping attributes."""
        DIM = args[3]
        nLinks = args[0][0]
        mass = args[0][1]
        pInertias = args[0][2]
        centreOfMass = args[0][2]
        rotOfPrincipalAxes = args[0][2]
        damping = args[0][1]
        dtype = args[1]
        requires_grad = args[2]
        monkeypatch.setattr(fct_robot, 'nLinks', nLinks, raising=True)
        monkeypatch.setattr(fct_robot, 'linksXdim', nLinks * DIM, raising=True)
        monkeypatch.setattr(fct_robot, 'dtype', dtype, raising=True)

        fct_robot.setInertialParams(mass, pInertias, centreOfMass, rotOfPrincipalAxes, damping, requires_grad=requires_grad)

        assert type(fct_robot.mass) is torch.nn.parameter.Parameter
        assert type(fct_robot.J1J2) is torch.nn.parameter.Parameter
        assert type(fct_robot.J1J2angle) is torch.nn.parameter.Parameter
        assert type(fct_robot.linkCoM) is torch.nn.parameter.Parameter
        assert type(fct_robot.rotationOfPrincipalAxes) is torch.nn.parameter.Parameter
        assert type(fct_robot.damping) is torch.nn.parameter.Parameter

        assert fct_robot.mass.requires_grad is requires_grad
        assert fct_robot.J1J2.requires_grad is requires_grad
        assert fct_robot.J1J2angle.requires_grad is requires_grad
        assert fct_robot.linkCoM.requires_grad is requires_grad
        assert fct_robot.rotationOfPrincipalAxes.requires_grad is requires_grad
        assert fct_robot.damping.requires_grad is requires_grad

        assert fct_robot.mass.equal(mass.to(dtype=dtype))
        assert fct_robot.J1J2.equal(pInertias[:, :-1].to(dtype=dtype))
        assert fct_robot.J1J2angle.allclose(fct_robot._computeAngle(pInertias).to(dtype=dtype))
        assert fct_robot.linkCoM.equal(centreOfMass.to(dtype=dtype))
        assert fct_robot.rotationOfPrincipalAxes.equal(rotOfPrincipalAxes.to(dtype=dtype))
        assert fct_robot.damping.equal(damping.to(dtype=dtype))

    @pytest.mark.parametrize('args',
                             itertools.product(zip(N_LINKS, TENSORS_3D), DTYPES, REQUIRES_GRAD, [DIM]),
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

    @pytest.mark.parametrize('device', DEVICES, ids=lambda device: f"{device}")
    def test_device(self, fct_robot, device):
        """The device property should return the torch.device of the parameters of the class."""
        fct_robot.staircase.data = fct_robot.staircase.to(device)
        expected = fct_robot.device
        assert expected == torch.device(device)

    @pytest.mark.parametrize('args',
                             itertools.product(zip(N_LINKS, TENSORS_3D, ROT_OF_P_AXES, ROT_MAT_INERTIA), DTYPES, [DIM]),
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
