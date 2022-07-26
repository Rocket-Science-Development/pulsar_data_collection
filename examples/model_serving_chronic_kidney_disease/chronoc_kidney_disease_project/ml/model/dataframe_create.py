# -*- coding: utf-8 -*-
import datetime

import pandas as pd


class Dframe:
    def __init__(self):
        pass

    # Function to convert prediction output to Pandas dataframe to be inserted in DB
    def dataframe_create(self, prediction, to_predict):
        # Creating dataframe with the output prediction
        pred_df = pd.DataFrame(prediction, columns=["class"])
        # Concat the input and output predicton dataframes on y-axis (columns)
        df = pd.concat([to_predict, pred_df], axis=1)
        # Adding current timestamp as a new column to existing Dataframe
        df.loc[:, "Timestamp"] = datetime.datetime.now()

        return df
