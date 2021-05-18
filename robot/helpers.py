import os
import time

import matplotlib.pyplot as plt
import torch


def generateRandomVectors(n, min, max, dtype):
    """Generate a set of random vectors of shape [n, len(min)] with values from [min, max].
    Arguments:
        n          - the number of vectors generated; int
        min        - minimum value for each element of the vector; iterable
        max        - maximum value for each element of the vector; iterable
        dtype      - data type of the returned vectors; torch.dtype
    Raises:
        ValueError - len(min) is not equal to len(max) or min > max for at least one element
    """
    if len(min) != len(max):
        raise ValueError(f"Different lengths of min and max. Expected len(min) == len(max); got len(min)={len(min)}, len(max)={len(max)}.")

    min = torch.tensor(min, dtype=dtype)
    max = torch.tensor(max, dtype=dtype)

    if not (min < max).all():
        raise ValueError(f"min < max not satisfied. Expected all True, got {(max > min)}.")

    range = max - min
    shape = torch.Size([n]) + range.shape
    vecs = torch.rand(shape, dtype=dtype) * range + min
    return vecs


def generateUnitVectors(shape, dtype):
    """Generate a set of random unit vectors of shape.
    Arguments:
        shape - the shape of the form (n, d) where n is number of vectors in the set and d is their dimension
        dtype - data type of the returned vectors; torch.dtype
    """
    vecs = 2 * torch.rand(shape, dtype=dtype) - 1
    vecs = torch.einsum('ni, n -> ni', vecs, (1 / vecs.norm(dim=1)))
    vecs.norm(dim=1).sum()
    return vecs


def printRunningLoss(running_loss, num_tenths, epoch=None):
    """Print the running loss every on the same line.
    Arguments:
        epoch        - epoch of the training; int
        running_loss - the loss to be printed; list
        num_tenths   - progress of the training in fraction of the whole; int in the interval [0, 10]
    """
    if epoch is not None:
        print(f"\repoch: {epoch}; average loss: {running_loss:.3e}, progress: [{'='*num_tenths + '-'*(10-num_tenths)}]", end='')
    else:
        print(f"\raverage loss: {running_loss:.3e}, progress: [{'='*num_tenths + '-'*(10-num_tenths)}]", end='')


def plotLoss(loss, offset=0):
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Training Loss')
    ax.plot(range(offset, len(loss)), loss[offset:])
    ax.set_xlabel('# of batches seen')
    ax.set_ylabel('avg. loss per batch')
    plt.yscale('log')
    plt.show()


def generateData(sim, cfg, set_size, batch_size, chunk_size=2**15):
    """Generate random data from simulator.
    Arguments:
        sim        - simulator used to generate the data
        cfg        - configuration file for a robot
        set_size   - size of the dataset
        batch_size - batch size argument for data loader
        chunk_size - the data is generated in chunks of size chunk_size (default=2**15)
    """
    gravity_direction = generateUnitVectors([set_size, 3], sim.dtype)
    thetas = generateRandomVectors(set_size, cfg['q_min'], cfg['q_max'], sim.dtype)
    dthetas = generateRandomVectors(set_size, cfg['dq_min'], cfg['dq_max'], sim.dtype)
    ddthetas = generateRandomVectors(set_size, cfg['ddq_min'], cfg['ddq_max'], sim.dtype)
    appliedTorques = torch.empty(thetas.shape, dtype=sim.dtype)

    with torch.no_grad():
        for b in range(0, set_size, chunk_size):
            e = b + chunk_size
            appliedTorques[b:e] = sim.getMotorTorque(thetas[b:e], dthetas[b:e], ddthetas[b:e], gravity_direction[b:e])

    dataset = torch.utils.data.TensorDataset(thetas, dthetas, ddthetas, appliedTorques, gravity_direction)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    return loader


def learningLoop(agent, optimiser, loader, losses):
    """Learning loop iterating over the whole dataset onece for the agent.
    Arguments:
        agent     - the learning agent
        optimiser - optimiser used in learning; assumes that it contains the agent parameters
        loader    - data loader containing the dataset
        losses    - list where average batch loses are appended
    Returns:
        avg_loss  - average loss of the whole dataset
    """
    tenthOfData = len(loader.dataset) // (loader.batch_size * 10)
    start = time.time()
    criterion = torch.nn.MSELoss(reduction='mean')
    avg_loss = 0.0
    for i, data in enumerate(loader, 1):
        theta = data[0].to(device=agent.device)
        dtheta = data[1].to(device=agent.device)
        ddtheta = data[2].to(device=agent.device)
        appliedTorque = data[3].to(device=agent.device)
        gravityDirection = data[4].to(device=agent.device)

        optimiser.zero_grad()
        appliedTorque_est = agent.getMotorTorque(theta, dtheta, ddtheta, directionOfGravity=gravityDirection)
        loss = criterion(appliedTorque_est, appliedTorque)
        loss.backward()
        optimiser.step()

        with torch.no_grad():
            avg_loss = avg_loss + (loss.item() - avg_loss) / i
            losses.append(loss.item())

        if (i % tenthOfData == 0):
            printRunningLoss(avg_loss, i // tenthOfData)
    print("")
    end = time.time()
    print(f"elapsed time {(end-start):.2f}s.")
    return avg_loss


def findFiles(path, files, prefixes=('.', '__')):
    """Find first occurance of files in path and its subdirectories.
    Arguments:
        path       - path to search
        files      - files to search for
        prefixes   - subdirectories with given prefixes will not be searched
    Returns:
        dictionary - key value pairs of found files and their paths
    """
    search_files = set(files)
    file_paths = {}
    for root, dirs, root_files in os.walk(path):
        tmp = [dir for dir in dirs if not dir.startswith(prefixes)]
        dirs.clear()
        dirs.extend(tmp)
        for file in search_files:
            if file in root_files:
                file_path = os.path.join(root, file)
                file_paths[file] = file_path
        tmp = [file for file in search_files if file not in file_paths.keys()]
        search_files = set(tmp)
    return file_paths


def findDirs(path, dirs, prefixes=('.', '__')):
    """Find first occurance of dirs in path and its subdirectories.
    Arguments:
        path       - path to search
        dirs       - directories to search for
        prefixes   - subdirectories with given prefixes will not be searched
    Returns:
        dictionary - key value pairs of found directories and their paths
    """
    search_dirs = set(dirs)
    dir_paths = {}
    if os.path.basename(path) in search_dirs:
        dir = os.path.basename(path)
        dir_paths[dir] = path
        search_dirs.remove(dir)
    for root, dirs, _ in os.walk(path):
        tmp = [dir for dir in dirs if not dir.startswith(prefixes)]
        dirs.clear()
        dirs.extend(tmp)
        for dir in search_dirs:
            if dir in dirs:
                dir_path = os.path.join(root, dir)
                dir_paths[dir] = dir_path
        tmp = [dir for dir in search_dirs if dir not in dir_paths.keys()]
        search_dirs = set(tmp)
    return dir_paths


def findProjectAndFilePaths(project, files, path=os.getcwd(), prefixes=('.', '__')):
    """Find the path of the project and paths of files in the project.
    Arguments:
        project   - project directory name to search
        files     - file names to search withing project directory
        path      - path to search for the project directory and the files
    Returns:
        tuple     - project path and dict of key value pairs of found directories and their paths
    Raises:
        NameError - when project is not found in the path
    """
    try:
        project_path = findDirs(path, [project], prefixes)[project]
    except KeyError:
        raise NameError(f"project {project} not found in {path}")

    file_paths = findFiles(project_path, files, prefixes)
    return project_path, file_paths
