############################################################################################
#
# Project:       Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:    Tensorflow-ALL-Segmentation-2020
# Project:       Tensorflow-ALL-Segmentation-2020
#
# Author:        Aniruddh Sharma
# Title:         Helper Class
# Description:   Helper functions for the Tensorflow-ALL-Segmentation-2020
# License:       MIT License
# Last Modified: 2021-01-17
#
############################################################################################

import logging, json, sys, time
import logging.handlers as handlers

from datetime import datetime


class Helpers():
    """ Helper Class

    Helper functions for the Tensorflow-ALL-Segmentation-2020.
    """

    def __init__(self, ltype, log=True):
        """ Initializes the Helpers Class. """

        # Loads system configs
        self.confs = {}
        self.loadConfs()

        # Sets system logging
        self.logger = logging.getLogger(ltype)
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        allLogHandler = handlers.TimedRotatingFileHandler(
            'Logs/all.log', when='H', interval=1, backupCount=0)
        allLogHandler.setLevel(logging.INFO)
        allLogHandler.setFormatter(formatter)

        errorLogHandler = handlers.TimedRotatingFileHandler(
            'Logs/error.log', when='H', interval=1, backupCount=0)
        errorLogHandler.setLevel(logging.ERROR)
        errorLogHandler.setFormatter(formatter)

        warningLogHandler = handlers.TimedRotatingFileHandler(
            'Logs/warning.log', when='H', interval=1, backupCount=0)
        warningLogHandler.setLevel(logging.WARNING)
        warningLogHandler.setFormatter(formatter)

        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(formatter)

        self.logger.addHandler(allLogHandler)
        self.logger.addHandler(errorLogHandler)
        self.logger.addHandler(warningLogHandler)
        self.logger.addHandler(consoleHandler)

        if log is True:
            self.logger.info("Helpers class initialization complete.")

    def loadConfs(self):
        """ Load the program configuration. """

        with open('confs.json') as confs:
            self.confs = json.loads(confs.read())
