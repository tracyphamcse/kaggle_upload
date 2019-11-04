import os
import time
# Create output folder
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

import logging
import sys
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG_FILE = "{}/logs.txt".format(out_dir)

def get_console_handler():
	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setFormatter(FORMATTER)
	return console_handler

def get_file_handler():
	file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight', encoding="utf-8")
	file_handler.setFormatter(FORMATTER)
	return file_handler

def get_logger(logger_name):
	logger = logging.getLogger(logger_name)

	logger.setLevel(logging.DEBUG) # better to have too much log than not enough

	logger.addHandler(get_console_handler())
	logger.addHandler(get_file_handler())

	# with this pattern, it's rarely necessary to propagate the error up to parent
	logger.propagate = True

	return logger
