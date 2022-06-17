# -*- coding: utf-8 -*-
import pickle as pkl

import pandas as pd
import sklearn

model = pkl.load(open("kidney_disease.pkl", "rb"))
test_data = pd.read_csv("test_data.csv")
