"""
by  initializing logging functionalities here in __init__.py
helps this module to be called whenever i am in src folder without
manually calling the module again and again and makes importing easier
""" 

import os
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
# asctime - when it happend
# levelname - how much impact is it creating or error type
# module - which module the error has raised
# message - the actual error message

log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir,exist_ok=True)


logging.basicConfig(
  level = logging.INFO,
  format= logging_str,
  handlers = [
    logging.FileHandler(log_filepath),
    logging.StreamHandler(sys.stdout)
  ]
)

logger = logging.getLogger("cnnClassifierLogger")