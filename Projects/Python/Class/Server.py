############################################################################################
#
# Project:       Peter Moss COVID-19 AI Research Project
# Repository:    Tensorflow-ALL-Segmentation-2020
# Project:       Tensorflow-ALL-Segmentation-2020
#
# Author:        Aniruddh Sharma
# Title:         Server Class
# Description:   Server functions for Tensorflow-ALL-Segmentation-2020
# License:       MIT License
# Last Modified: 2020-06-29
#
############################################################################################

import jsonpickle

import numpy as np

from flask import Flask, request, Response

from Classes.Helpers import Helpers
from Classes.Model import Model

class Server():
    """ Server helper class
    Server functions for the COVID-19 Tensorflow DenseNet Classifier.
    """

    def __init__(self, model):
        """ Initializes the class. """

        self.Helpers = Helpers("Server", False)

        self.model = model

    def start(self):
        """ Starts the server. """

        app = Flask(__name__)

        @app.route('/Inference', methods=['POST'])
        def Inference():
            """ Responds to standard HTTP request. """

            message = ""
            classification, confidence = self.model.http_classify(request)

            if classification == 1:
                message = "Leukemia detected!"
                diagnosis = "Positive"
            elif classification == 0:
                message = "Leukemia not detected!"
                diagnosis = "Negative"

            resp = jsonpickle.encode({
                'Response': 'OK',
                'Message': message,
                'Diagnosis': diagnosis,
                'Confidence': np.asscalar(confidence)
            })

            return Response(response=resp, status=200, mimetype="application/json")

        app.run(host=self.Helpers.confs["server"]["ip"],
                port=self.Helpers.confs["server"]["port"])
