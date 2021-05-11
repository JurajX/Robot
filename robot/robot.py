import torch


class Robot(torch.nn.Module):
    """docstring for Robot."""

    # CONSTANTS
    G = 9.80665
    DIM = 3
    # generators of so(3) Lie algebra of shape (3, 3, 3), where [i, :, :] corresponds to i-th generator
    so3GEN = [[[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]], [[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]], [[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]]]

    def __init__(self, nLinks, directionOfGravity, rotationAxesOfJoints, frameCoordinates, dtype=torch.float32):
        """Creates an instance of a Robot.
        Arguments:
            nLinks               - number of links (int)
            directionOfGravity   - unit vector indicating the direction of the gravitational acceleration; iterable of shape (3)
            rotationAxesOfJoints - coordinates of the pseudo-vectors parametrising the rotation axes of joints; iterable of shape (nLinks, 3)
            frameCoordinates     - coordinates of the frame i relative to the frame i-1, i ∈ {1, ..., nLinks} ; iterable of shape (nLinks, 3)
            dtype                - type of underlying data; string, torch.float32 or torch.float64
        """
        super(Robot, self).__init__()

        self.nLinks = nLinks
        self.linksXdim = self.nLinks * self.DIM
        self.dtype = self._formatDtype(dtype)
        self.SO3GEN = self._makeParameter(torch.tensor(self.so3GEN, dtype=self.dtype))

        self.staircase = self._makeStaircase()
        self.staircaseIdentity = self._createStaircaseIdentity()

        # A is a block diagonal containing
        # B is a nilpotent containing
        # C is a staircaseIdentity
        self.makeBlockRotMat = eval(f"lambda A, B, C: A{'.matmul(C + B'*(self.nLinks - 1)} {')'*(self.nLinks - 1)}")

        self.gravAccel = None
        self.setGravAccel(directionOfGravity)
        tmp = self._makeTensor(rotationAxesOfJoints, shape=(self.nLinks, self.DIM))
        tmp /= tmp.norm(dim=1, keepdim=True)
        self.rotationAxesOfJoints = self._makeParameter(tmp, (self.nLinks, self.DIM))
        tmp = torch.einsum('ni, ijk -> njk', self.rotationAxesOfJoints, self.SO3GEN)
        self.L_rotationAxesOfJoints = self._makeParameter(tmp, (self.nLinks, self.DIM, self.DIM))
        self.frameCoordinates = self._makeParameter(frameCoordinates, (self.nLinks, self.DIM))
        self.triuFrameCoordinates = self._makeTriuFrameCoordinates()

        self.mass = self._makeRandParameter(shape=(self.nLinks, ), requires_grad=True)
        self.J1J2 = self._makeRandParameter(shape=(self.nLinks, self.DIM - 1), requires_grad=True, abs=True)
        self.J1J2angle = self._makeRandParameter(shape=(self.nLinks, 1), requires_grad=True)
        self.linkCoM = self._makeRandParameter(shape=(self.nLinks, self.DIM), requires_grad=True)
        self.rotationOfPrincipalAxes = self._makeRandParameter(shape=(self.nLinks, self.DIM), requires_grad=True)
        self.damping = self._makeRandParameter(shape=(self.nLinks, ), requires_grad=True)

    @staticmethod
    def _checkShape(tensor, shape):
        """Check if the given tensor is of the given shape.
        Arguments:
            tensor     - a tensor; torch.Tensor
            shape      - an iterable indicating the shape
        Returns:
            None
        Raises:
            ValueError - the given tensor does not have the desired shape
        """
        if tensor.shape != torch.Size(shape):
            raise ValueError(f"Incorrect shape of the data. Expected {torch.Size(shape)} got {tensor.shape}.")

    @staticmethod
    def _formatDtype(dtype):
        """Format the given dtype into a torch.dtype.
        Arguments:
            dtype       - type of underlying data; string, torch.float32 or torch.float64
        Returns:
            torch.dtype - torch.float32 or torch.float64
        Raises:
            TypeError   - an incorrect dtype was provided
        """
        if (dtype == 'float32') or (dtype == 'torch.float32'):
            return torch.float32
        elif (dtype == 'float64') or (dtype == 'torch.float64'):
            return torch.float64
        elif (dtype == torch.float32) or (dtype == torch.float64):
            return dtype
        else:
            raise TypeError(f"Incorrect dtype. Expected torch.float32 or torch.float64 but got {dtype}.")

    def _makeTensor(self, data, device='cpu', shape=None):
        """Create a tensor from the given data and check if it has appropriate shape.
        Arguments:
            data         - data to be given to the created parameter; iterable
            shape        - an iterable indicating the shape of the parameter
        Returns:
            torch.Tensor - containing the given data
        Raises:
            ValueError   - the given data does not have the desired shape
        """
        if isinstance(data, torch.Tensor):
            tensor = data.to(device=device, dtype=self.dtype, non_blocking=True)
        else:
            tensor = torch.tensor(data, dtype=self.dtype, device=device)
        if shape is not None:
            self._checkShape(tensor, shape)
        return tensor

    def _makeParameter(self, data, shape=None, device='cpu', requires_grad=False):
        """Create a parameter from the given data and check if it has appropriate shape.
        Arguments:
            data               - data to be given to the created parameter; iterable
            shape              - an iterable indicating the shape of the parameter
            requires_grad      - determines if the parameter is trainable; boolean (default=False)
        Returns:
            torch.nn.Parameter - containing the given data
        Raises:
            ValueError         - the given data does not have the desired shape
        """
        tensor = self._makeTensor(data, device=device, shape=shape)
        return torch.nn.Parameter(tensor, requires_grad=requires_grad)

    def _makeRandParameter(self, shape, requires_grad=False, abs=False):
        """Create a parameter of the given shape, and populates it with random data from [0, 1).
        Arguments:
            shape              - an iterable indicating the shape of the parameter
            requires_grad      - determines if the parameter is trainable; boolean (default=False)
            abs                - determines if the absolute value is applied to the randomly generated values; boolean (default=False)
        Returns:
            torch.nn.Parameter - containing random data from [0, 1)
        """
        if abs:
            tmp = torch.rand(shape, dtype=self.dtype).abs()
        else:
            tmp = torch.rand(shape, dtype=self.dtype)
        return torch.nn.Parameter(tmp, requires_grad=requires_grad)

    def _makeStaircase(self):
        """Create a block diagonal parameter containing [[1., 1., 1.]] on the diagonal.
        Arguments:
            None
        Returns:
            torch.nn.Parameter - containing the staircase
        Raises:
            None
        """
        shape = (self.linksXdim, self.nLinks)
        ones = torch.ones(self.nLinks, self.DIM, 1)
        tmp = torch.unbind(ones, dim=0)
        staircase = torch.block_diag(*tmp)
        staircase = self._makeParameter(staircase, shape)
        return staircase

    def _makeTriuStaircase(self):
        """Create a tensor of shape (nLinks+1, nLinks*3) where:
            * the slice [0, :] containes ones, and
            * the slice [1:, :] is a block upper triangular tensor with the non-zero blocks being [1., 1., 1.].
        Arguments:
            None
        Returns:
            torch.Tensor - containing the described data
        Raises:
            None
        """
        ones = torch.ones(self.nLinks, self.nLinks)
        triu = torch.triu(ones)
        tmp = triu.unsqueeze(-1).expand(-1, -1, self.DIM).reshape(self.nLinks, self.linksXdim)
        ones = torch.ones(1, self.linksXdim)
        triuStaircase = torch.cat((ones, tmp), dim=0)
        return triuStaircase

    def _createStaircaseIdentity(self):
        """Create a zero parameter of shape (1, nLinks+1, nLinks*3, nLinks*3) except for the diagonal slices [0, :, range(nLinks*3), range(nLinks*3)].
        These slices are  of shape (nLinks+1, nLinks*3) and are equal to the tensor returned by the member function _makeTriuStaircase.
        Arguments:
            None
        Returns:
            torch.nn.Parameter - containing the described data
        Raises:
            None
        """
        shape = (1, self.nLinks + 1, self.linksXdim, self.linksXdim)
        staircaseIdentity = torch.zeros(shape)
        staircaseIdentity[0, :, range(self.linksXdim), range(self.linksXdim)] = self._makeTriuStaircase()
        staircaseIdentity = self._makeParameter(staircaseIdentity, shape)
        return staircaseIdentity

    def _makeTriuFrameCoordinates(self):
        """Create a block upper triangular parameter (without the diagonal) of shape (nLinks*3, nLinks) containing the i-th frame coordinate
        in the i-th row.
        Arguments:
            None
        Returns:
            torch.nn.Parameter - containing the frame coordinates in the upper triangular blocks
        Raises:
            None
        """
        shape = (self.linksXdim, self.nLinks)
        ones = torch.ones(self.DIM, self.nLinks, self.nLinks)
        tmp = torch.triu(ones, diagonal=1).transpose(0, 1).reshape(shape)
        triuFrameCoordinates = tmp * self.frameCoordinates.reshape(self.linksXdim, 1)
        triuFrameCoordinates = self._makeParameter(triuFrameCoordinates, shape)
        return triuFrameCoordinates

    @staticmethod
    def _computeAngle(principalInertias):
        """Compute the angle between the firs two, i.e. (x, y)-axes, principal inertias from the law of cosines.
        Arguments:
            principalInertias - the main principal inertias of each link; iterable of shape (nLinks, 3)
        Returns:
            torch.Tensor      - containing the angle between the firs two principal inertias
        """
        square = principalInertias.square()
        mask = torch.tensor([1., 1., -1])
        numerator = (square * mask).sum(dim=1)
        denominator = torch.tensor(2.) * principalInertias[:, :2].prod(dim=1)
        angle = torch.acos(numerator / denominator).unsqueeze(-1)
        return angle

    def _computeJ3(self):
        """Compute the third, i.e. z-axis, principal inertia from the first two and the angle between them from the law of cosines.
        Arguments:
            principalInertias - the main principal inertias of each link; iterable of shape (nLinks, 3)
        Returns:
            torch.Tensor      - containing the third principal inertia
        """
        J1J2sq = self.J1J2.square().sum(dim=1)
        J1xJ2 = self.J1J2.prod(dim=1).abs()
        cos = self.J1J2angle.squeeze().cos()
        two = torch.tensor(2., dtype=self.dtype)
        J3sq = J1J2sq - (two*cos*J1xJ2)
        J3 = J3sq.sqrt()
        return J3

    def setGravAccel(self, directionOfGravity, requires_grad=False):
        """Set a custom parameter for gravitational acceleration and check if it has an appropriate shape.
        Arguments:
            directionOfGravity - value describing the direction of gravitational acceleration; iterable
            shape              - an iterable indicating the shape of the parameter
            requires_grad      - determines if the parameter is trainable; boolean (default=False)
        Returns:
            torch.nn.Parameter - containing gravitational acceleration vector with the norm equal to 9.80665
        Raises:
            ValueError         - the given directionOfGravity does not have the desired shape
        """
        shape = (self.DIM, )
        param = self._makeParameter(directionOfGravity, shape, requires_grad=requires_grad)
        param.data *= (self.G / param.norm())
        self.gravAccel = param

    def setPrincipalInertias(self, principalInertias, requires_grad=False):
        """Set custom parameters for principal inertis and check if it has an appropriate shape.
        Arguments:
            principalInertias  - the main principal inertias of each link; iterable of shape (nLinks, 3)
            requires_grad      - determines if the parameters are trainable; boolean (default=False)
        Returns:
            None
        Raises:
            ValueError         - the given data does not have the desired shape
        """
        J1J2J3 = self._makeTensor(principalInertias, device=self.device, shape=(self.nLinks, self.DIM))
        J1J2angle = self._computeAngle(J1J2J3)
        self.J1J2 = self._makeParameter(J1J2J3[:, :2], (self.nLinks, self.DIM - 1), device=self.device, requires_grad=requires_grad)
        self.J1J2angle = self._makeParameter(J1J2angle, (self.nLinks, 1), device=self.device, requires_grad=requires_grad)

    def setRotOfPrincipalAxes(self, rotOfPrincipalAxes, requires_grad=False):
        """Set custom parameters for principal inertias and check if it has an appropriate shape.
        Arguments:
            rotOfPrincipalAxes - coordinates of the pseudo-vectors describing the rotation of principal inertias; iterable of shape (nLinks, 3)
            requires_grad      - determines if the parameters are trainable; boolean (default=False)
        Returns:
            None
        Raises:
            ValueError         - the given data does not have the desired shape
        """
        self.rotationOfPrincipalAxes = self._makeParameter(rotOfPrincipalAxes, (self.nLinks, self.DIM),
                                                           device=self.device,
                                                           requires_grad=requires_grad)

    def setMass(self, mass, requires_grad=False):
        """Set custom parameters for masses of links and check if it has an appropriate shape.
        Arguments:
            mass          - mass of each link; iterable of shape (nLinks, )
            requires_grad - determines if the parameters are trainable; boolean (default=False)
        Returns:
            None
        Raises:
            ValueError    - the given data does not have the desired shape
        """
        self.mass = self._makeParameter(mass, (self.nLinks, ), device=self.device, requires_grad=requires_grad)

    def setCoM(self, centreOfMass, requires_grad=False):
        """Set custom parameters for centre of masses of links and check if it has an appropriate shape.
        Arguments:
            centreOfMass  - the centre of mass of the link i relative to the frame i-1, i ∈ {1, ..., nLinks}; iterable of shape (nLinks, 3)
            requires_grad - determines if the parameters are trainable; boolean (default=False)
        Returns:
            None
        Raises:
            ValueError    - the given data does not have the desired shape
        """
        self.linkCoM = self._makeParameter(centreOfMass, (self.nLinks, self.DIM), device=self.device, requires_grad=requires_grad)

    def setDamping(self, damping, requires_grad=False):
        """Set custom parameters for damping of links and check if it has an appropriate shape.
        Arguments:
            damping       - damping coefficient at each joint; iterable of shape (nLinks, )
            requires_grad - determines if the parameters are trainable; boolean (default=False)
        Returns:
            None
        Raises:
            ValueError    - the given data does not have the desired shape
        """
        self.damping = self._makeParameter(damping, (self.nLinks, ), device=self.device, requires_grad=requires_grad)

    def setInertialParams(self, mass, principalInertias, centreOfMass, rotOfPrincipalAxes, damping, requires_grad=False):
        """Set custom inertial parameters for the robot class.
        Arguments:
            mass               - mass of each link; iterable of shape (nLinks, )
            principalInertias  - the main principal inertias of each link; iterable of shape (nLinks, 3)
            centreOfMass       - the centre of mass of the link i relative to the frame i-1, i ∈ {1, ..., nLinks}; iterable of shape (nLinks, 3)
            rotOfPrincipalAxes - coordinates of the pseudo-vectors describing the rotation of principal inertias; iterable of shape (nLinks, 3)
            damping            - damping coefficient at each joint; iterable of shape (nLinks, )
            requires_grad      - determines if the parameters are trainable; boolean (default=False)
        Returns:
            None
        Raises:
            ValueError         - the given data does not have the desired shape
        """
        self.setMass(mass, requires_grad=requires_grad)
        self.setPrincipalInertias(principalInertias, requires_grad=requires_grad)
        self.setCoM(centreOfMass, requires_grad=requires_grad)
        self.setRotOfPrincipalAxes(rotOfPrincipalAxes, requires_grad=requires_grad)
        self.setDamping(damping, requires_grad=requires_grad)

    @property
    def device(self):
        """Return the device of the parameters of the class.
        Returns:
            torch.device - indicating the device of the parameters
        """
        return self.staircase.device

    @property
    def principalAxesInertia(self):
        """
        Return the principal inertias of the links.
        Returns:
            torch.Tensor - containing the principal inertias of each link
        """
        J3 = self._computeJ3()
        return torch.cat((self.J1J2.abs(), J3.view(-1, 1)), dim=1)

    @property
    def inertia(self):
        """
        Return the inertia tensor of each link in the link's frame.
        Returns:
            torch.Tensor - containing the inertia tensors of each link
        """
        principalAxesInertia = self.principalAxesInertia
        diagInertia = torch.diag_embed(principalAxesInertia)
        L_rotationOfPrincipalAxes = torch.einsum('ni, ijk -> njk', self.rotationOfPrincipalAxes, self.SO3GEN)
        principalRot = torch.matrix_exp(L_rotationOfPrincipalAxes)
        rotatedInertia = principalRot.matmul(diagInertia).matmul(principalRot.transpose(-1, -2))
        return rotatedInertia

    def _makeBigRotMat(self, Rot, dRot):
        """
        Create a tensor of shape (batch_size, nLinks+1, nLinks*3, nLinks*3), where, for each batch entry:
            * The part [:, 0, :, :] forms a block upper triangular tensor of shape (nLinks, nLinks), with each [3, 3] block being a coordinate
              transform from (j+1)-th fram to the (i)-th frame for i ≥ j and i, j ∈ {0, ..., nLinks-1} where [i, j] is a slice of (nLinks, nLinks).
            * The slices [:, i, :, :] represent partial derivatives of the slice [:, 0, :, :] w.r.t the i-th joint angle for i ∈ {1, ..., nLinks}.
        Arguments:
            Rot          - batch tensor of the coo. transforms from i-th frame to (i-1)-th frame for all i ∈ {1, ..., nLinks}; i.e. (R_1, ... R_n)
            dRot         - batch tensor of partial derivatives of the above w.r.t the joint angles; i.e. ((d_1)(R_1), (d_2)(R_2), ..., (d_n)(R_n))
        Returns:
            torch.Tensor - containing the big rotation tensor as described above
        """
        batch_size = Rot.shape[0]
        BDiag = torch.zeros((batch_size, self.nLinks, self.DIM, self.nLinks, self.DIM), dtype=self.dtype, device=self.device)

        # place rotation matrices on the block diagonal
        BDiag[:, range(self.nLinks), :, range(self.nLinks)] = Rot.transpose(0, 1)
        # make space for dRot and replace Rot by dRot in the respective leayers
        BDiag = BDiag.unsqueeze(1).repeat(1, self.nLinks + 1, 1, 1, 1, 1)
        BDiag[:, range(1, 1 + self.nLinks), range(self.nLinks), :, range(self.nLinks), :] = dRot.transpose(0, 1)

        BDiag = BDiag.reshape(batch_size, self.nLinks + 1, self.linksXdim, self.linksXdim)
        Nilptnt = torch.zeros_like(BDiag)
        Nilptnt[:, :, :-self.DIM, self.DIM:] = BDiag[:, :, self.DIM:, self.DIM:]

        # adjust the differentials so that only the rows BDiag[:, i, :3*i, :] are non-zero i ∈ {1, ..., nLinks}
        BDiag[:, 1:] = self.staircaseIdentity[:, 1:].flip(dims=(1, 2, 3)).matmul(BDiag[:, 1:])
        bigRot = self.makeBlockRotMat(BDiag, Nilptnt, self.staircaseIdentity)
        return bigRot

    def _makeRho(self, bigRot):
        """
        Create a tensor of shape (batch_size, nLinks+1, 3, nLinks, 3), where
            * [:, 0 , :, 0 , :] are 3x3 identity matrices,
            * [:, 1:, :, 0 , :] are zeros, and
            * [:, : , :, 1:, :] is bigRot[:, :, :3, :-3] (after reshaping).
        Arguments:
            bigRot       - big rotation tensor as described in the member function _makeBigRotMat
        Returns:
            torch.Tensor - containing the data as described above
        """
        batch_size = bigRot.shape[0]
        zeros = torch.zeros((batch_size, self.nLinks + 1, self.DIM, self.DIM), dtype=self.dtype, device=self.device)
        extract = bigRot[:, :, :self.DIM, :-self.DIM]
        rho = torch.cat((zeros, extract), dim=-1)
        rho = rho.reshape(batch_size, self.nLinks + 1, self.DIM, self.nLinks, self.DIM)
        rho[:, 0, :, 0, :] = torch.eye(self.DIM, dtype=self.dtype, device=self.device)
        return rho

    def _makeCentreOfMassCoordinates(self, bigRot):
        """
        Create a block upper triangular tensor of shape (batch_size, nLinks + 1, nLinks, 3, nLinks), where:
            * The part [:, 0, i, :, j] gives the centre of mass coordinates of the (j+1)-th link expressed in the i-th frame for i ≥ j and
            i, j ∈ {0, ..., nLinks-1}.
            * The slices [:, i, :, :, :] represent partial derivatives of the slice [:, 0, :, :] w.r.t the i-th joint angle for i ∈ {1, ..., nLinks}.
        Arguments:
            bigRot       - big rotation tensor as described in the member function _makeBigRotMat
        Returns:
            torch.Tensor - containing the data as described above
        """
        batch_size = bigRot.shape[0]
        tmp = self.staircase * self.linkCoM.view(-1, 1)
        tmp = tmp + self.triuFrameCoordinates
        CentreOfMassCoordinates = bigRot.matmul(tmp)
        CentreOfMassCoordinates = CentreOfMassCoordinates.reshape(batch_size, self.nLinks + 1, self.nLinks, self.DIM, self.nLinks)
        return CentreOfMassCoordinates

    def _makeMassMatrix(self, bigRot, rho, centreOfMassCoordinates):
        """
        Create a tensor of shape (batch_size, nLinks + 1, nLinks, nLinks), where:
            * The part [:, 0, :, :] is the 'mass matrix' of the robot.
            * The slices [:, i, :, :] represent the partial derivatives of the 'mass matrix' w.r.t. the i-th joint angle for i ∈ {1, ..., nLinks}.
        Arguments:
            batch_size              - batch size
            bigRot                  - big rotation tensor as described in the member function _makeBigRotMat
            rho                     - rho tensor as described in the member function _makeRho
            centreOfMassCoordinates - coordinates of the centre of mass as described in the member function _makeCentreOfMassCoordinates
        Returns:
            torch.Tensor            - containing the data as described above
        """
        batch_size = bigRot.shape[0]
        # create massMatIner
        tmp = bigRot.reshape(batch_size, self.nLinks + 1, self.nLinks, self.DIM, self.nLinks, self.DIM)
        dR_qh = torch.einsum('bdnimj, ni -> bdmjn', tmp, self.rotationAxesOfJoints)
        massMatIner = torch.einsum('bnjm, njk, bdnko -> bdmo', dR_qh[:, 0], self.inertia, dR_qh)
        massMatIner[:, 1:] = massMatIner[:, 1:] + massMatIner[:, 1:].transpose(-1, -2)

        # create L_c
        tmp = torch.einsum('ijk, bdlnj -> bdlnik', self.SO3GEN, rho)
        tmp1 = torch.einsum('bdnim, belnik -> bdemlnk', centreOfMassCoordinates, tmp)
        Lc = tmp1[:, :, 0]
        Lc[:, 1:] = Lc[:, 1:] + tmp1[:, 0, 1:]

        # create massMatTran
        Lc_qh = torch.einsum('bdmlnk, nk -> bdmln', Lc, self.rotationAxesOfJoints)
        massMatTran = torch.einsum('bmin, m, bdmio -> bdno', Lc_qh[:, 0], self.mass, Lc_qh)
        massMatTran[:, 1:] = massMatTran[:, 1:] + massMatTran[:, 1:].transpose(-1, -2)

        return massMatIner + massMatTran

    def _make_EoM_parameters(self, theta, directionOfGravity=None):
        """
        Create mass matrix, Christoffel symbols, gravity torque, and potential energy of the robot.
        Arguments:
            theta                 - joint angles; iterable of shape (batch_size, nLinks)
            directionOfGravity    - gravity direction for each sample in the base frame of the robot; iterable of shape (batch_size, 3)
        Returns:
            tuple of torch.Tensor - containing the data as described above
        """
        theta = self._makeTensor(theta, device=self.device)
        batch_size = theta.shape[0]

        L_q = self.L_rotationAxesOfJoints.unsqueeze(0) * theta.view(batch_size, self.nLinks, 1, 1)
        Rot = torch.matrix_exp(L_q)
        dRot = self.L_rotationAxesOfJoints.matmul(Rot)

        bigRot = self._makeBigRotMat(Rot, dRot)
        rho = self._makeRho(bigRot)
        centreOfMassCoordinates = self._makeCentreOfMassCoordinates(bigRot)
        massMat = self._makeMassMatrix(bigRot, rho, centreOfMassCoordinates)

        tmp = massMat[:, 1:]
        christoffelSymbols = 0.5 * (tmp.permute(0, 2, 3, 1) + tmp.permute(0, 2, 1, 3) - tmp)   # free index is the first

        tmp = centreOfMassCoordinates[:, :, 0, :, :]
        if directionOfGravity is not None:
            gravAccel = self._makeTensor(directionOfGravity, device=self.device)
            gravAccel = self.G * gravAccel / gravAccel.norm(dim=1).unsqueeze(-1)
            tmp = tmp.matmul(self.mass)
            potEnergy = -torch.einsum('bdi, bi -> bd', tmp, gravAccel)
        else:
            potEnergy = -tmp.matmul(self.mass).matmul(self.gravAccel)

        return massMat[:, 0], christoffelSymbols, potEnergy[:, 1:], potEnergy[:, 0]

    def getAngularAcceleration(self, theta, dtheta, motorTorque=None, directionOfGravity=None):
        """
        Compute resulting angular acceleration of the robot links based on the given joint angles, angular velocities and motor torques.
        Arguments:
            theta              - joint angles; iterable of shape (batch_size, nLinks)
            dtheta             - joint angular velocities; iterable of shape (batch_size, nLinks)
            motorTorque        - torques applied by the motors; iterable of shape (batch_size, nLinks)
            directionOfGravity - gravity direction for each sample in the base frame of the robot; iterable of shape (batch_size, 3)
        Returns:
            torch.Tensor       - containing the computed angular accelerations
        """
        theta = self._makeTensor(theta, device=self.device)
        dtheta = self._makeTensor(dtheta, device=self.device)
        if motorTorque is None:
            motorTorque = torch.zeros(1, dtype=self.dtype, device=self.device)
        else:
            motorTorque = self._makeTensor(motorTorque, device=self.device)

        massMat, christoffelSymbols, gravityTorque, _ = self._make_EoM_parameters(theta, directionOfGravity)
        massMatInv = massMat.inverse()

        christoffelTorque = torch.einsum('bn, bmno, bo -> bm', dtheta, christoffelSymbols, dtheta)
        frictionTorque = dtheta.mul(self.damping.abs())

        torque = motorTorque - christoffelTorque - gravityTorque - frictionTorque
        angularAcceleration = torch.einsum('bmn, bn -> bm', massMatInv, torque)
        return angularAcceleration

    def getMotorTorque(self, theta, dtheta, ddtheta, directionOfGravity=None):
        """
        Compute needed motor torques of the robot links based on the provided joint angles, angular velocities and angular accelerations.
        Arguments:
            theta              - joint angles; iterable of shape (batch_size, nLinks)
            dtheta             - joint angular velocities; iterable of shape (batch_size, nLinks)
            ddtheta            - joint angular accelerations; iterable of shape (batch_size, nLinks)
            directionOfGravity - gravity direction for each sample in the base frame of the robot; iterable of shape (batch_size, 3)
        Returns:
            torch.Tensor       - containing the computed angular accelerations
        """
        theta = self._makeTensor(theta, device=self.device)
        dtheta = self._makeTensor(dtheta, device=self.device)
        ddtheta = self._makeTensor(ddtheta, device=self.device)

        massMat, christoffelSymbols, gravityTorque, _ = self._make_EoM_parameters(theta, directionOfGravity)
        inertiaTorque = torch.einsum('bmn, bn -> bm', massMat, ddtheta)
        christoffelTorque = torch.einsum('bn, bmno, bo -> bm', dtheta, christoffelSymbols, dtheta)
        frictionTorque = dtheta.mul(self.damping.abs())
        return inertiaTorque + christoffelTorque + gravityTorque + frictionTorque

    def getLagrangian(self, theta, dtheta, directionOfGravity=None):
        """
        Compute the Lagrangian of the robot.
        Arguments:
            theta              - joint angles; iterable of shape (batch_size, nLinks)
            dtheta             - joint angular velocities; iterable of shape (batch_size, nLinks)
            directionOfGravity - gravity direction for each sample in the base frame of the robot; iterable of shape (batch_size, 3)
        Returns:
            torch.Tensor       - containing the computed Lagrangian of the robot
        """
        theta = self._makeTensor(theta, device=self.device)
        dtheta = self._makeTensor(dtheta, device=self.device)

        massMat, _, _, potEnergy = self._make_EoM_parameters(theta, directionOfGravity)
        kinEnergy = 0.5 * torch.einsum('bm, bmn, bn -> b', dtheta, massMat, dtheta)
        return kinEnergy - potEnergy
