import os


def files_in_dir(directory, exts=['.jpg', '.png'], include_dir=False, sort=True):
    '''returns filenames with exts in directory.'''
    if not os.path.isdir(directory):
        print('Directory {} not exist!'.format(directory))
        return None
    
    if isinstance(exts, str):
        exts = [exts]

    fns = [f for f in os.listdir(directory) if os.path.splitext(f)[1] in exts]
    if sort:
        fns = sorted(fns)
    if include_dir:
        fns = [os.path.join(directory, fn) for fn in fns]
    return fns


def files_by_text(txt_file, root_dir=None, sort=True):
    '''Given a txt file, whose each line specifys a data path.
    '''
    if not os.path.isfile(txt_file):
        print('File {} not exist!'.format(txt_file))
        return None
    with open(txt_file, 'r') as fp:
        fns = fp.read()
        fns = fns.split('\n')
        fns = [fn.strip() for fn in fns]
        fns = [fn for fn in fns if len(fn) > 0]
        fns = [fn for fn in fns if fn[0] != '#']
        if sort:
            fns = sorted(fns)
        if root_dir is not None:
            fns = [os.path.join(root_dir, fn) for fn in fns]
        return fns
