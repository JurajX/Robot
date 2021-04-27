import pytest
import torch
import robot.robot as robot
import itertools

from tests.gradient import gradient

# =========== TEST CONSTANTS ===========
N_LINKS = [1, 3, 5, 7]
BATCH_SIZES = [1, 8, 16, 32]

DTYPES = [torch.float32, torch.float64]
DEVICES = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']

so3gen = [[[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]], [[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]], [[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]]]
SO3GEN = torch.tensor(so3gen)

DIRECTIONS = [torch.tensor([0., 0., 5.]), [-3., 0., 0.], (1., 1., 0.)]
tmp = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [-1., 0., 0.], [0., -1., 0.], [0., 0., -1.], [0., 1., 0.]])
ROT_AXES = [tmp[:n] for n in N_LINKS]

PI = 4 * torch.atan(torch.tensor(1.0, dtype=torch.float64))
THETAS = torch.tensor([2 * PI, PI, PI / 2, PI / 3, PI / 4, PI / 6, 0.0])
D_THETAS = torch.tensor([1., 2., 3., 4., 5., 6., 7.])
DD_THETAS = torch.tensor([10., 20., 30., 40., 50., 60., 70.])

tmp = [[0., 0., 0.1], [0., 0., 0.1], [0., 0., 0.1], [0., 0., 0.1], [0., 0., 0.1], [0., 0., 0.1], [0., 0., 0.1]]
FRAME_COORDINATES = [tmp[:n] for n in N_LINKS]


# =========== FIXTURES ===========
@pytest.fixture(
    scope='function',
    params=itertools.product(zip(N_LINKS, ROT_AXES, FRAME_COORDINATES), DIRECTIONS, DTYPES, DEVICES),
    ids=lambda fixture_value: f"n:{fixture_value[0][0]}-{type(fixture_value[1]).__name__}-f{str(fixture_value[2])[-2:]}-{fixture_value[3]}")
def param_robot(request):
    params = request.param
    nLinks = params[0][0]
    directionOfGravity = params[1]
    rotationAxesOfJoints = params[0][1]
    frameCoordinates = params[0][2]
    dtype = params[2]
    device = params[3]
    rbt = robot.Robot(nLinks, directionOfGravity, rotationAxesOfJoints, frameCoordinates, dtype)
    return rbt.to(device=device)


# =========== TESTING ===========


class Test_Computation():

    @pytest.mark.parametrize('theta', THETAS, ids=lambda theta: f"{theta/PI:.2f}pi")
    def test_makeBigRotMat(self, param_robot, theta):
        """The part [:, 0, :, :] of the _makeBigRotMat member function output should form a block upper triangular tensor of shape (nLinks, nLinks),
        with each [3, 3] block being the proper coordinate transform."""
        L_qh = param_robot.L_rotationAxesOfJoints.unsqueeze(0)
        DIM = param_robot.DIM
        linksXdim = param_robot.linksXdim
        dtype = param_robot.dtype
        device = param_robot.device
        theta = theta.to(device=device)
        sin = L_qh
        cos = L_qh.matmul(L_qh.transpose(-1, -2))
        id = torch.eye(L_qh.shape[-1], dtype=dtype, device=device).expand(L_qh.shape[0], -1, -1) - cos
        Rot = id + sin * torch.sin(theta) + cos * torch.cos(theta)
        dRot = L_qh.matmul(Rot)

        expected = torch.zeros((linksXdim, linksXdim), dtype=dtype, device=device)
        for r in range(0, linksXdim, DIM):
            tmp = torch.eye(DIM, dtype=dtype, device=device)
            for c in range(0, linksXdim, DIM):
                if r > c:
                    continue
                tmp = tmp.matmul(Rot[0, int(c / 3)])
                expected[r:r + 3, c:c + 3] = tmp
        returned = param_robot._makeBigRotMat(Rot, dRot)[0, 0]
        assert returned.allclose(expected)

    @pytest.mark.parametrize('theta', THETAS, ids=lambda theta: f"{theta/PI:.2f}pi")
    def test_makeBigRotMat_exp(self, param_robot, theta):
        """The part [:, 0, :, :] of the _makeBigRotMat member function output should form a block upper triangular tensor of shape (nLinks, nLinks),
        with each [3, 3] block being the proper coordinate transform."""
        L_qh = param_robot.L_rotationAxesOfJoints.unsqueeze(0)
        DIM = param_robot.DIM
        linksXdim = param_robot.linksXdim
        dtype = param_robot.dtype
        device = param_robot.device
        theta = theta.to(device=device)
        L_q = L_qh * theta.to(dtype=dtype).unsqueeze(-1).unsqueeze(-1)
        Rot = torch.matrix_exp(L_q)
        dRot = L_qh.matmul(Rot)

        expected = torch.zeros((linksXdim, linksXdim), dtype=dtype, device=device)
        for r in range(0, linksXdim, DIM):
            tmp = torch.eye(DIM, dtype=dtype, device=device)
            for c in range(0, linksXdim, DIM):
                if r > c:
                    continue
                tmp = tmp.matmul(Rot[0, int(c / 3)])
                expected[r:r + 3, c:c + 3] = tmp
        returned = param_robot._makeBigRotMat(Rot, dRot)[0, 0]
        assert returned.allclose(expected)

    def test_makeBigRotMat_diff(self, param_robot, thetas=THETAS):
        """The slices [:, i, :, :] of the _makeBigRotMat member function output should represent partial derivatives of the slice [:, 0, :, :]
        w.r.t the i-th joint angle.
        Assumes that test_makeBigRotMat and test_makeBigRotMat_exp pass!"""
        dtype = param_robot.dtype
        nLinks = param_robot.nLinks
        device = param_robot.device
        theta = thetas[:nLinks].unsqueeze(0).to(dtype=dtype)
        theta = theta.to(device=device)
        theta.requires_grad_(True)

        L_qh = param_robot.L_rotationAxesOfJoints.unsqueeze(0)
        L_q = L_qh * theta.unsqueeze(-1).unsqueeze(-1)
        Rot = torch.matrix_exp(L_q)
        dRot = L_qh.matmul(Rot)
        bigRot = param_robot._makeBigRotMat(Rot, dRot)

        returned = bigRot[:, 1:]
        expected = gradient(bigRot[:, 0].sum(dim=0), theta, create_graph=False, strict=True).permute(2, 3, 0, 1)
        if dtype == torch.float64:
            assert returned.allclose(expected)
        else:
            assert returned.allclose(expected, rtol=1e-04, atol=1e-05)

    @pytest.mark.parametrize('bSize', BATCH_SIZES, ids=lambda bSize: f"batch:{bSize}")
    def test_makeRho(self, param_robot, bSize):
        """The _makeRho member function should return a tensor of shape (batch_size, nLinks+1, 3, nLinks, 3) where
            * [:, 0 , :, 0 , :] are 3x3 identity matrices,
            * [:, 1:, :, 0 , :] are zeros, and
            * [:, : , :, 1:, :] is bigRot[:, :, :3, :-3] (after reshaping)."""
        DIM = param_robot.DIM
        nLinks = param_robot.nLinks
        linksXdim = param_robot.linksXdim
        dtype = param_robot.dtype
        device = param_robot.device
        bigRot = torch.rand((bSize, nLinks + 1, linksXdim, linksXdim), dtype=dtype, device=device)

        rho = param_robot._makeRho(bigRot)
        id = torch.eye(DIM, dtype=dtype, device=device).unsqueeze(0).expand(bSize, -1, -1)
        rho[:, 0, :, 0, :].equal(id)
        zeros = torch.zeros_like(rho[:, 1:, :, 0, :])
        rho[:, 1:, :, 0, :].equal(zeros)
        modBigRot = bigRot[:, :, :3, :-3].reshape(bSize, nLinks + 1, DIM, nLinks - 1, DIM)
        assert rho[:, :, :, 1:, :].equal(modBigRot)

    @pytest.mark.parametrize('bSize', BATCH_SIZES, ids=lambda bSize: f"batch:{bSize}")
    def test_makeCoM_Coos(self, param_robot, bSize):
        """The _makeCentreOfMassCoordinates member function should return a tensor of shape (bSize, nLinks + 1, nLinks, 3, nLinks), with the part
        [:, 0, i, :, j] giving the centre of mass coordinates of the (j+1)-th link expressed in the i-th frame."""
        DIM = param_robot.DIM
        nLinks = param_robot.nLinks
        linksXdim = param_robot.linksXdim
        dtype = param_robot.dtype
        device = param_robot.device

        bigRot = torch.rand((bSize, nLinks + 1, linksXdim, linksXdim), dtype=dtype, device=device)
        tmp = torch.zeros((nLinks, DIM, nLinks), dtype=dtype, device=device)

        for r in range(0, nLinks):
            for c in range(0, nLinks):
                if r > c:
                    continue
                elif r == c:
                    tmp[r, :, c] = param_robot.linkCoM[r]
                else:
                    tmp[r, :, c] = param_robot.frameCoordinates[r]
        tmp = tmp.reshape(linksXdim, nLinks)

        expected = bigRot.matmul(tmp).reshape(bSize, nLinks + 1, nLinks, DIM, nLinks)
        returned = param_robot._makeCentreOfMassCoordinates(bigRot)
        assert returned.equal(expected)

    @pytest.mark.parametrize('bSize', BATCH_SIZES, ids=lambda bSize: f"batch:{bSize}")
    def test_makeMassMatrix(self, param_robot, bSize, so3gen=SO3GEN):
        """The _makeMassMatrix member function should return a tensor of shape (batch_size, nLinks + 1, nLinks, nLinks), with the part
        [:, 0, :, :] being the 'mass matrix' of the robot."""
        DIM = param_robot.DIM
        nLinks = param_robot.nLinks
        linksXdim = param_robot.linksXdim
        dtype = param_robot.dtype
        device = param_robot.device

        bigRot = torch.rand((bSize, nLinks + 1, linksXdim, linksXdim), dtype=dtype, device=device)
        rho = torch.rand((bSize, nLinks + 1, DIM, nLinks, DIM), dtype=dtype, device=device)
        centreOfMassCoordinates = torch.rand((bSize, nLinks + 1, nLinks, DIM, nLinks), dtype=dtype, device=device)

        # create massMatIner
        tmp = bigRot.reshape((bSize, nLinks + 1, nLinks, DIM, nLinks, DIM))
        dR_qh = torch.einsum('bdnimj, ni -> bdmjn', tmp, param_robot.rotationAxesOfJoints)
        massMatIner = torch.einsum('bnjm, njk, bdnko -> bdmo', dR_qh[:, 0], param_robot.inertia, dR_qh)[:, 0]

        # create L_c
        tmp = torch.einsum('ijk, bdlnj -> bdlnik', so3gen.to(dtype=dtype, device=device), rho)
        tmp1 = torch.einsum('bdnim, belnik -> bdemlnk', centreOfMassCoordinates, tmp)
        Lc = tmp1[:, 0, 0]

        # create massMatTran
        Lc_qh = torch.einsum('bmlnk, nk -> bmln', Lc, param_robot.rotationAxesOfJoints)
        massMatTran = torch.einsum('bmin, m, bmio -> bno', Lc_qh, param_robot.mass, Lc_qh)
        expected = massMatIner + massMatTran
        returned = param_robot._makeMassMatrix(bigRot, rho, centreOfMassCoordinates)[:, 0]
        assert returned.allclose(expected, rtol=1e-04, atol=1e-05)

    def test_makeMassMatrix_diff(self, param_robot, thetas=THETAS):
        """The _makeMassMatrix member function should return a tensor of shape (batch_size, nLinks + 1, nLinks, nLinks), with the slices
        [:, i, :, :] represent the partial derivatives of the 'mass matrix' w.r.t. the i-th joint angle.
        Assumes that test_makeMassMatrix, test_makeCoM_Coos, test_makeRho, and test_makeBigRotMat_diff pass!"""
        nLinks = param_robot.nLinks
        dtype = param_robot.dtype
        device = param_robot.device
        theta = thetas[:nLinks].unsqueeze(0).to(dtype=dtype)
        theta = theta.to(device=device)
        theta.requires_grad_(True)

        L_qh = param_robot.L_rotationAxesOfJoints.unsqueeze(0)
        L_q = L_qh * theta.unsqueeze(-1).unsqueeze(-1)
        Rot = torch.matrix_exp(L_q)
        dRot = L_qh.matmul(Rot)
        bigRot = param_robot._makeBigRotMat(Rot, dRot)
        rho = param_robot._makeRho(bigRot)
        centreOfMassCoordinates = param_robot._makeCentreOfMassCoordinates(bigRot)
        massMat = param_robot._makeMassMatrix(bigRot, rho, centreOfMassCoordinates)

        returned = massMat[:, 1:]
        expected = gradient(massMat[:, 0].sum(dim=0), theta, create_graph=False, strict=True).permute(2, 3, 0, 1)

        if dtype == torch.float64:
            assert returned.allclose(expected)
        else:
            assert returned.allclose(expected, rtol=1e-04, atol=1e-03)

    def test_make_EoM_parameters(self, param_robot, thetas=THETAS):
        """The _make_EoM_parameters member function should return the proper mass matrix, Christoffel symbols, gravity torque, and
        potential energy of the robot.
        Assumes that all the above test functions pass!"""
        nLinks = param_robot.nLinks
        dtype = param_robot.dtype
        device = param_robot.device
        theta = thetas[:nLinks].unsqueeze(0).to(dtype=dtype)
        theta = theta.to(device=device)

        L_qh = param_robot.L_rotationAxesOfJoints.unsqueeze(0)
        L_q = L_qh * theta.unsqueeze(-1).unsqueeze(-1)
        Rot = torch.matrix_exp(L_q)
        dRot = L_qh.matmul(Rot)
        bigRot = param_robot._makeBigRotMat(Rot, dRot)
        rho = param_robot._makeRho(bigRot)
        centreOfMassCoordinates = param_robot._makeCentreOfMassCoordinates(bigRot)
        massMat = param_robot._makeMassMatrix(bigRot, rho, centreOfMassCoordinates)
        christoffelSymbols = 0.5 * (massMat[:, 1:].permute(0, 2, 3, 1) + massMat[:, 1:].permute(0, 2, 1, 3) - massMat[:, 1:])
        potEnergy = -centreOfMassCoordinates[:, :, 0, :, :].matmul(param_robot.mass).matmul(param_robot.gravAccel)

        returned = param_robot._make_EoM_parameters(theta)
        assert returned[0].allclose(massMat[:, 0])
        assert returned[1].allclose(christoffelSymbols)
        assert returned[2].allclose(potEnergy[:, 1:])
        assert returned[3].allclose(potEnergy[:, 0])

    def test_torque(self, param_robot, thetas=THETAS, dthetas=D_THETAS, ddthetas=DD_THETAS):
        """The getMotorTorque member function should return the same tensor as numerically calculated Euler-Lagrange equations."""
        nLinks = param_robot.nLinks
        dtype = param_robot.dtype
        device = param_robot.device
        theta = thetas[:nLinks].unsqueeze(0).to(dtype=dtype)
        theta = theta.to(device=device)
        theta.requires_grad_(True)
        dtheta = dthetas[:nLinks].unsqueeze(0).to(dtype=dtype)
        dtheta = dtheta.to(device=device)
        dtheta.requires_grad_(True)
        ddtheta = ddthetas[:nLinks].unsqueeze(0).to(dtype=dtype)
        ddtheta = ddtheta.to(device=device)

        Lagrangian = param_robot.getLagrangian(theta, dtheta)
        L_qdot = gradient(Lagrangian.sum(dim=0), dtheta, create_graph=True, strict=True)
        L_q = gradient(Lagrangian.sum(dim=0), theta, create_graph=False, strict=True)
        L_qdot_qdot = gradient(L_qdot.sum(dim=0), dtheta, create_graph=False, strict=True).permute(1, 0, 2)
        L_qdot_q = gradient(L_qdot.sum(dim=0), theta, create_graph=False, strict=True).permute(1, 0, 2)

        inertiaTorque = torch.einsum('bij, bj -> bi', L_qdot_qdot, ddtheta)
        coriolisTorque = torch.einsum('bij, bj -> bi', L_qdot_q, dtheta)
        frictionTorque = dtheta.mul(param_robot.damping)

        expected = inertiaTorque + coriolisTorque - L_q + frictionTorque
        returned = param_robot.getMotorTorque(theta, dtheta, ddtheta)
        if dtype == torch.float64:
            assert returned.allclose(expected)
        else:
            assert returned.allclose(expected, rtol=1e-02, atol=1e-02)

    def test_acceleration(self, param_robot, thetas=THETAS, dthetas=D_THETAS):
        """The getAngularAcceleration member function should return the same tensor as numerically calculated accelerations from the Euler-Lagrange
        equations."""
        nLinks = param_robot.nLinks
        dtype = param_robot.dtype
        device = param_robot.device
        theta = thetas[:nLinks].unsqueeze(0).to(dtype=dtype)
        theta = theta.to(device=device)
        theta.requires_grad_(True)
        dtheta = dthetas[:nLinks].unsqueeze(0).to(dtype=dtype)
        dtheta = dtheta.to(device=device)
        dtheta.requires_grad_(True)

        Lagrangian = param_robot.getLagrangian(theta, dtheta)
        L_qdot = gradient(Lagrangian.sum(dim=0), dtheta, create_graph=True, strict=True)
        L_q = gradient(Lagrangian.sum(dim=0), theta, create_graph=False, strict=True)
        L_qdot_qdot = gradient(L_qdot.sum(dim=0), dtheta, create_graph=False, strict=True).permute(1, 0, 2)
        L_qdot_q = gradient(L_qdot.sum(dim=0), theta, create_graph=False, strict=True).permute(1, 0, 2)

        coriolisTorque = torch.einsum('bij, bj -> bi', L_qdot_q, dtheta)
        frictionTorque = dtheta.mul(param_robot.damping)
        expected = torch.einsum('bij, bj -> bi', L_qdot_qdot.inverse(), (L_q - coriolisTorque - frictionTorque))
        coriolisTorque - L_q + frictionTorque
        returned = param_robot.getAngularAcceleration(theta, dtheta)
        if dtype == torch.float64:
            assert returned.allclose(expected)
        else:
            assert returned.allclose(expected, rtol=1e-02, atol=1e-02)
