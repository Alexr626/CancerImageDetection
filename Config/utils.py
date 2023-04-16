import os
def is_dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)