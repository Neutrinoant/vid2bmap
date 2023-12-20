import logging
from logging import getLogger
from logging.handlers import SysLogHandler
from logging import FileHandler, StreamHandler, Formatter

import os
import sys
from pathlib import Path
from datetime import datetime
from pytz import timezone

from typing import Callable

# ==============< Configurations >==============

DEFAULT_LOG_ADDRESS = ''        # host:port
DEFAULT_LOG_PERIOD = '1000'     # positive integer
DEFAULT_LOG_LEVEL = 'DEBUG'     # DEBUG/INFO/WARNING/ERROR/CRITICAL

DEFAULT_LOG_NAME = 'basic'      # log identifier discriminating multiple log objects

WRITE_TO_CONSOLE = True         # write to terminal if true
CONSOLE_LOG_LEVEL = 'DEBUG'      # DEBUG/INFO/WARNING/ERROR/CRITICAL

WRITE_TO_FILE = True            # write to file if true
FILE_LOG_LEVEL = 'DEBUG'        # DEBUG/INFO/WARNING/ERROR/CRITICAL
FILE_LOG_DIR = "../../logs"     # relative dir to this file

# ==============================================

class Logger(object):
    def __init__(self, name, console=True, logdir:str=None, tz='Asia/Seoul'):
        self.logger = getLogger(name)
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical
        self.period = int(os.getenv('LOG_PERIOD', DEFAULT_LOG_PERIOD))
        self.now = datetime.now(timezone(tz)).strftime("%Y-%m-%d-%H-%M-%S-%f-%Z")       # current time indicator

        level = getattr(logging, os.getenv('LOG_LEVEL', DEFAULT_LOG_LEVEL).upper())
        address = os.getenv('LOG_ADDRESS', DEFAULT_LOG_ADDRESS)
        if address:
            host, port = address.split(':')
            file_name = os.path.basename(sys.argv[0])
            log_message = Formatter(file_name+' %(message)s')
            syslogHandler = SysLogHandler(address=(host, int(port)))
            syslogHandler.setFormatter(log_message)
            self.logger.addHandler(syslogHandler)

        basic_message = Formatter('%(asctime)s - %(levelname)s - %(message)s')

        if console:
            streamHandler = StreamHandler(sys.stdout)
            streamHandler.setFormatter(basic_message)
            streamHandler.setLevel(CONSOLE_LOG_LEVEL)
            self.logger.addHandler(streamHandler)

        if logdir:
            # make log directory if not exist
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            # add filehandler
            self.filepath = Path(logdir) / Path(f"{name}_{self.now}.log")
            file_handler = FileHandler(self.filepath)
            file_handler.setFormatter(basic_message)
            file_handler.setLevel(FILE_LOG_LEVEL)
            self.logger.addHandler(file_handler)

        self.logger.setLevel(level)

    def periodic(self, testCount, numOfTests, message):
        func = self.debug if testCount % self.period else self.info
        func('Test {} out of {}: {}'.format(testCount, numOfTests, message))

    def change_path(self, new_path):
        # remove existing file handlers
        fhdrls = [hdrl for hdrl in self.logger.handlers if isinstance(hdrl, FileHandler)]
        if not fhdrls:
            self.logger.warning('no file handler.')
            return
        formatter = fhdrls[0].formatter
        # clear existing log file
        with open(new_path, 'wt') as f: pass
        for hdrl in fhdrls:
            self.logger.removeHandler(hdrl)
        # add new file handler
        new_handler = FileHandler(new_path)
        new_handler.setFormatter(formatter)
        self.logger.addHandler(new_handler)


__LOG_DIR__ = (Path(__file__) / Path(FILE_LOG_DIR)).resolve() if WRITE_TO_FILE else None


# ===============< EXPORTED REGION >================

'''
    Usage:
        from logger import logger
        logger.debug("test!")
        logger.info("test!")
        logger.warning("test!")
        logger.error("test!")
        logger.critical("test!")
'''
logger = Logger(DEFAULT_LOG_NAME, console=WRITE_TO_CONSOLE, logdir=__LOG_DIR__)

# ==================================================


def checkpoint(f: Callable):
    # helper decorator: check function start and end
    def wrapper(*args, **kwargs):
        logger.info(f"======== start of {f.__name__}() ========")
        output = f(*args, **kwargs)
        logger.info(f"======== end of {f.__name__}() ========")
        return output
    return wrapper