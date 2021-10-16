import itertools

import pytest
import torch
import yaml

import src.robot as robot
import src.utils.find_paths as paths

cfg_name = 'test_config.yml'
proj_name = 'Robot'
_, config_path = paths.findProjectAndFilePaths(proj_name, [cfg_name])
with open(config_path[cfg_name], "r") as ymlfile:
    tmp = yaml.safe_load(ymlfile)
    cfg = tmp['test_init_consts']
    panda = tmp['panda']

for key, value in cfg.items():
    cfg[key] = eval(value)


@pytest.fixture(scope='class')
def cls_robot():
    rbt = robot.Robot(panda['nLinks'], panda['directionOfGravity'], panda['rotationAxesOfJoints'], panda['frameCoordinates'], panda['dtype'])
    return rbt


class Test_RobotInitialisers():

    @pytest.mark.parametrize('shape', cfg['shapes'], ids=lambda shape: f"shape:{shape}")
    def test_checkShape(self, cls_robot, shape):
        """The _checkShape member function should not throw when the shape of the tensor and the given shape agree."""
        tensor = torch.rand(shape)
        try:
            cls_robot._checkShape(tensor, shape)
        except ValueError:
            pytest.fail(f'The function failed for shape {shape}.')

    @pytest.mark.parametrize('shape', cfg['shapes'], ids=lambda shape: f"shape:{shape}")
    def test_checkShape_raises(self, cls_robot, shape):
        """The _checkShape member function should raise an error when the shape of the tensor and the given shape do not agree."""
        tensor = torch.rand(shape + (1, ))
        with pytest.raises(ValueError) as except_info:
            cls_robot._checkShape(tensor, shape)
        assert except_info.value.args[0] == f"Incorrect shape of the data. Expected {torch.Size(shape)} got {tensor.shape}."

    @pytest.mark.parametrize('dtype', cfg['all_dtypes'], ids=lambda dtype: f"{'str_'+dtype if type(dtype) is str else str(dtype)[-7:]}")
    def test_formatDtype(self, cls_robot, dtype):
        """The _formatDtype member function should return torch.float32 or torch.float64."""
        returned = cls_robot._formatDtype(dtype)
        assert (returned is torch.float32) or (returned is torch.float64)
        assert str(returned)[-2:] == str(dtype)[-2:]

    @pytest.mark.parametrize('dtype', cfg['bad_dtypes'], ids=lambda dtype: f"dtype:{dtype}")
    def test_formatDtype_raises(self, cls_robot, dtype):
        """The _formatDtype member function should raise TypeError when provided improper dtype."""
        with pytest.raises(TypeError) as except_info:
            cls_robot._formatDtype(dtype)
        assert except_info.value.args[0] == f"Incorrect dtype. Expected torch.float32 or torch.float64 but got {dtype}."

    @pytest.mark.parametrize('args', itertools.product(cfg['data'], cfg['dtypes']), ids=lambda args: f"{type(args[0]).__name__}-f{str(args[1])[-2:]}")
    def test_makeTensor(self, cls_robot, monkeypatch, args):
        """The _makeTensor member function should return a Tensor containing the given data of dtype associated with the Robot class."""
        data = args[0]
        dtype = args[1]
        monkeypatch.setattr(cls_robot, 'dtype', dtype, raising=True)

        if isinstance(data, torch.Tensor):
            expected = data.to(dtype=dtype, non_blocking=True)
        else:
            expected = torch.tensor(data, dtype=dtype)
        returned = cls_robot._makeTensor(data)
        assert returned.equal(expected)

    @pytest.mark.parametrize('args',
                             itertools.product(cfg['r_grads'], cfg['dtypes']),
                             ids=lambda args: f"r_grad:{'T' if args[0] else 'F'}-f{str(args[1])[-2:]}")
    def test_makeParameter(self, cls_robot, monkeypatch, args):
        """The _makeParameter member function should return a Parameter of proper dtype and with properly set requires_grad flag."""
        requires_grad = args[0]
        dtype = args[1]
        data = [[1., 2., 3.], [4., 5., 6.]]
        monkeypatch.setattr(cls_robot, 'dtype', dtype, raising=True)

        expected = torch.nn.Parameter(torch.tensor(data, dtype=dtype), requires_grad=requires_grad)
        returned = cls_robot._makeParameter(data, requires_grad=requires_grad)
        assert type(expected) == type(returned)
        assert type(expected).requires_grad == type(returned).requires_grad
        assert returned.equal(expected)

    @pytest.mark.parametrize('args', itertools.product(cfg['shapes'], cfg['dtypes']), ids=lambda args: f"shape:{args[0]}-f{str(args[1])[-2:]}")
    def test_makeRandnParameter(self, cls_robot, monkeypatch, args):
        """The _makeRandnParameter member function should return a random Parameter of a proper shape and dtype."""
        shape = args[0]
        dtype = args[1]
        monkeypatch.setattr(cls_robot, 'dtype', dtype, raising=True)

        returned = cls_robot._makeRandnParameter(shape)
        assert type(returned) is torch.nn.parameter.Parameter
        assert returned.shape == shape
        assert returned.dtype == dtype

    @pytest.mark.parametrize('args', itertools.product(cfg['shapes'], cfg['dtypes']), ids=lambda args: f"shape:{args[0]}-f{str(args[1])[-2:]}")
    def test_makeZeroParameter(self, cls_robot, monkeypatch, args):
        """The _makeZeroParameter member function should return a random Parameter of a proper shape and dtype."""
        shape = args[0]
        dtype = args[1]
        monkeypatch.setattr(cls_robot, 'dtype', dtype, raising=True)

        returned = cls_robot._makeZeroParameter(shape)
        assert type(returned) is torch.nn.parameter.Parameter
        assert returned.shape == shape
        assert returned.dtype == dtype
        assert returned.equal(torch.zeros(shape, dtype=dtype))

    @pytest.mark.parametrize('args', itertools.product(cfg['n_links'], [cfg['dim']]), ids=lambda args: f"nLinks:{args[0]}")
    def test_makeStaircase(self, cls_robot, monkeypatch, args):
        """The _makeStaircase member function should return a staircase Parameter of a proper shape."""
        nLinks = args[0]
        DIM = args[1]
        linksXdim = nLinks * DIM
        monkeypatch.setattr(cls_robot, 'nLinks', nLinks, raising=True)
        monkeypatch.setattr(cls_robot, 'DIM', DIM, raising=True)
        monkeypatch.setattr(cls_robot, 'linksXdim', linksXdim, raising=True)

        expected = torch.block_diag(*([torch.ones(DIM, 1)] * nLinks))
        returned = cls_robot._makeStaircase()
        assert type(returned) is torch.nn.parameter.Parameter
        assert returned.equal(expected)

    @pytest.mark.parametrize('args', itertools.product(cfg['n_links'], [cfg['dim']]), ids=lambda args: f"nLinks:{args[0]}")
    def test_makeTriuStaircase(self, cls_robot, monkeypatch, args):
        """The _makeTriuStaircase member function should return a triuStaircase tensor of a proper shape."""
        nLinks = args[0]
        DIM = args[1]
        linksXdim = nLinks * DIM
        monkeypatch.setattr(cls_robot, 'nLinks', nLinks, raising=True)
        monkeypatch.setattr(cls_robot, 'DIM', DIM, raising=True)
        monkeypatch.setattr(cls_robot, 'linksXdim', linksXdim, raising=True)

        expected1stRow = torch.ones(linksXdim)
        triu = torch.triu(torch.ones(nLinks, nLinks))
        expectedRest = triu.unsqueeze(-1).expand(-1, -1, DIM).reshape(nLinks, linksXdim)
        returned = cls_robot._makeTriuStaircase()
        assert type(returned) is torch.Tensor
        assert returned.shape == (nLinks + 1, linksXdim)
        assert returned[0].equal(expected1stRow)
        assert returned[1:].equal(expectedRest)

    @pytest.mark.parametrize('args', itertools.product(cfg['n_links'], [cfg['dim']]), ids=lambda args: f"nLinks:{args[0]}")
    def test_createStaircaseIdentity(self, cls_robot, monkeypatch, args):
        """The _createStaircaseIdentity member function should return a staircaseIdentity Parameter of a proper shape."""
        nLinks = args[0]
        DIM = args[1]
        linksXdim = nLinks * DIM
        monkeypatch.setattr(cls_robot, 'nLinks', nLinks, raising=True)
        monkeypatch.setattr(cls_robot, 'DIM', DIM, raising=True)
        monkeypatch.setattr(cls_robot, 'linksXdim', linksXdim, raising=True)

        triu = torch.triu(torch.ones(nLinks, nLinks))
        triuStaircase = torch.ones((nLinks + 1, linksXdim))
        triuStaircase[1:] = triu.unsqueeze(-1).expand(-1, -1, DIM).reshape(nLinks, linksXdim)

        returned = cls_robot._createStaircaseIdentity()
        assert type(returned) is torch.nn.parameter.Parameter
        assert returned.shape == (1, nLinks + 1, linksXdim, linksXdim)
        assert returned[0, :, range(linksXdim), range(linksXdim)].equal(triuStaircase)
        assert returned[:, :, range(linksXdim), range(linksXdim)].sum() == triuStaircase.sum()

    @pytest.mark.parametrize('args', itertools.product(cfg['n_links'], [cfg['dim']]), ids=lambda args: f"nLinks:{args[0]}")
    def test_makeTriuFrameCoordinates(self, cls_robot, monkeypatch, args):
        """The _makeTriuFrameCoordinates member function should return a triuFrameCoordinates Parameter of a proper shape."""
        nLinks = args[0]
        DIM = args[1]
        linksXdim = nLinks * DIM
        frameCoordinates = torch.nn.Parameter(torch.rand(nLinks + 1, DIM), requires_grad=False)
        monkeypatch.setattr(cls_robot, 'nLinks', nLinks, raising=True)
        monkeypatch.setattr(cls_robot, 'DIM', DIM, raising=True)
        monkeypatch.setattr(cls_robot, 'linksXdim', linksXdim, raising=True)
        monkeypatch.setattr(cls_robot, 'frameCoordinates', frameCoordinates, raising=True)

        tmp = torch.triu(torch.ones(DIM, nLinks, nLinks), diagonal=1).transpose(0, 1).reshape(linksXdim, nLinks)
        expected = tmp * cls_robot.frameCoordinates[1:].reshape(linksXdim, 1)  # do not include the fram from origin to the first joint

        returned = cls_robot._makeTriuFrameCoordinates()
        assert type(returned) is torch.nn.parameter.Parameter
        assert returned.shape == (linksXdim, nLinks)
        assert returned.equal(expected)
