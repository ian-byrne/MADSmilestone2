import pandas as pd
import numpy as np
import io
import torch
import requests
from PIL import Image


def load_data():
    """Load stata data from S3, rounds 1 - 10; contains diagnoses, scoring, subject ID to map to images
    hc1disescn9 : 1 - YES to dementia/Alzheimers, 2 - NO Dementia, may want to drop -9 and -1?, may need to relabel 7.
    cg1dclkdraw: score of drawing
    spid: Subject ID, number maps to number in image file names"""

    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    round_data = pd.DataFrame()

    for val in values:
        data = pd.io.stata.read_stata(
            "https://clockdrawingbattery.s3.us-west-1.amazonaws.com/NHATS_Round_"
            + str(val)
            + "_SP_File.dta"
        )
        data = data[
            ["spid", "cg" + str(val) + "dclkdraw", "hc" + str(val) + "disescn9"]
        ]
        data["round"] = val

        # Rename columns
        data.rename(
            columns={
                "cg" + str(val) + "dclkdraw": "cg" + str(int(val / val)) + "dclkdraw",
                "hc" + str(val) + "disescn9": "hc" + str(int(val / val)) + "disescn9",
            },
            inplace=True,
        )
        round_data = round_data.append(data)

    return round_data


def load_images():
    """Currently loading one image at a time and turning the bool matrix into the inverse
    numpy array"""
    url = "https://clockdrawingimages1.s3.us-west-1.amazonaws.com/10000003.tif"

    response = requests.get(url)  # , stream = True)
    f = io.BytesIO(response.content)
    im = Image.open(f)
    imarray = np.logical_not(np.array(im)).astype(int)
    return imarray


def hats_load_data():
    """ This loads in the data with all the columns needed for label creation using the
    NHATs Dementia Classification criteria. It loads diagnosis variables, ID, round number,
    and Cognitive test variables for the domains of Orientation, Memory and Executive Functioning."""

    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    round_data = pd.DataFrame()

    for val in values:
        data = pd.io.stata.read_stata(
            "https://clockdrawingbattery.s3.us-west-1.amazonaws.com/NHATS_Round_"
            + str(val)
            + "_SP_File.dta"
        )

        if val == 1:
            data = data[
                [
                    "spid",
                    "cg" + str(val) + "dclkdraw",
                    "hc" + str(val) + "disescn9",
                    "cg" + str(val) + "presidna1",
                    "cg" + str(val) + "presidna3",
                    "cg" + str(val) + "vpname1",
                    "cg" + str(val) + "vpname3",
                    "cg" + str(val) + "todaydat1",
                    "cg" + str(val) + "todaydat2",
                    "cg" + str(val) + "todaydat3",
                    "cg" + str(val) + "todaydat4",
                    "cg" + str(val) + "dwrdimmrc",
                    "cg" + str(val) + "dwrddlyrc",
                ]
            ]
        else:
            data = data[
                [
                    "spid",
                    "cg" + str(val) + "dclkdraw",
                    "hc" + str(val) + "disescn9",
                    "cp" + str(val) + "dad8dem",
                    "cg" + str(val) + "presidna1",
                    "cg" + str(val) + "presidna3",
                    "cg" + str(val) + "vpname1",
                    "cg" + str(val) + "vpname3",
                    "cg" + str(val) + "todaydat1",
                    "cg" + str(val) + "todaydat2",
                    "cg" + str(val) + "todaydat3",
                    "cg" + str(val) + "todaydat4",
                    "cg" + str(val) + "dwrdimmrc",
                    "cg" + str(val) + "dwrddlyrc",
                ]
            ]

        data["round"] = val

        # Rename columns
        data.rename(
            columns={
                "cg" + str(val) + "dclkdraw": "cg" + str(int(val / val)) + "dclkdraw",
                "hc" + str(val) + "disescn9": "hc" + str(int(val / val)) + "disescn9",
                "cp" + str(val) + "dad8dem": "cp" + str(int(val / val)) + "dad8dem",
                "cg" + str(val) + "presidna1": "cg" + str(int(val / val)) + "presidna1",
                "cg" + str(val) + "presidna3": "cg" + str(int(val / val)) + "presidna3",
                "cg" + str(val) + "vpname1": "cg" + str(int(val / val)) + "vpname1",
                "cg" + str(val) + "vpname3": "cg" + str(int(val / val)) + "vpname3",
                "cg" + str(val) + "todaydat1": "cg" + str(int(val / val)) + "todaydat1",
                "cg" + str(val) + "todaydat2": "cg" + str(int(val / val)) + "todaydat2",
                "cg" + str(val) + "todaydat3": "cg" + str(int(val / val)) + "todaydat3",
                "cg" + str(val) + "todaydat4": "cg" + str(int(val / val)) + "todaydat4",
                "cg" + str(val) + "dwrdimmrc": "cg" + str(int(val / val)) + "dwrdimmrc",
                "cg" + str(val) + "dwrddlyrc": "cg" + str(int(val / val)) + "dwrddlyrc",
            },
            inplace=True,
        )
        round_data = round_data.append(data)

    return round_data


def load_np_files(data, target):
    """ Takes in filenames for image data and target data
    expands the dimensions for the image data to reflect grayscale
    of dimension 1, prepping for the dataloader, and then zips the
    image data and labels together for future dataloading. Also returns
    label tensors for each split"""

    # Get data
    x_data = np.load(data)
    y_data = np.load(target)

    # Need to add that extra dimension for grayscale depth of 1 channel
    x_data = np.expand_dims(x_data, 1)
    print(x_data.shape)

    # Zip image data and labels together
    data = [(x, y) for x, y in zip(x_data, y_data)]

    # turn targets into tensors
    y_tensor = torch.from_numpy(y_data)

    return data, y_tensor

