# -*- coding: utf-8 -*-
import csv
import logging
import pickle as pkl
from io import StringIO

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


def start_end_decorator(func):
    def wrapper():
        print("Start")
        func()
        print("End")


def print_name():
    print("Alex")


print_name()
