import os.path as osp
import logging


def get_logger(config, logname):
    fh = logging.FileHandler(osp.join(config.log.output_dir, logname), 'w')
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    log = logging.getLogger('root')
    log.setLevel(logging.INFO)
    log.handlers = []       # No duplicated handlers
    log.propagate = False   # workaround for duplicated logs in ipython
    log.addHandler(ch)
    log.addHandler(fh)
    return log
