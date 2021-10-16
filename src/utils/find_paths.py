import os


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
                file_paths[file] = os.path.join(root, file)
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
    for root, dirs, _ in os.walk(path):
        tmp = [dir for dir in dirs if not dir.startswith(prefixes)]
        dirs.clear()
        dirs.extend(tmp)
        for dir in search_dirs:
            if dir in dirs:
                dir_paths[dir] = os.path.join(root, dir)
        tmp = [dir for dir in search_dirs if dir not in dir_paths.keys()]
        search_dirs = set(tmp)
    for dir in search_dirs:
        if dir in path:
            root, _ = path.split(dir)
            dir_paths[dir] = os.path.join(root, dir)
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
