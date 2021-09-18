import numpy as np


def clean_data(rounds_df):

    # Clean the row values in hc1disescn9 , cg1dclkdraw
    # hc1disescn9 asks if subject has dementia or alzheimers: 1 YES, 2 NO

    df = rounds_df
    hc_items = [
        ("2 NO", 2),
        (" 2 NO", 2),
        ("1 YES", 1),
        (" 1 YES", 1),
        ("-9 Missing", np.nan),
        ("-8 DK", np.nan),
        ("7 PREVIOUSLY REPORTED", 7),
        ("-1 Inapplicable", np.nan),
        ("-7 RF", np.nan),
    ]

    cg_items = [
        ("-2 Proxy says cannot ask SP", np.nan),
        ("-7 SP refused to draw clock", np.nan),
        ("-4 SP did not attempt to draw clock", np.nan),
        ("-3 Proxy says can ask SP but SP unable to answer", np.nan),
        ("-1 Inapplicable", np.nan),
        ("-9 Missing", np.nan),
    ]

    for item in hc_items:
        df.hc1disescn9.replace(item[0], item[1], inplace=True)

    # Remove cg1dclkdraw subjects that did not draw a clock, or image data is missing
    for item in cg_items:
        df.cg1dclkdraw.replace(item[0], item[1], inplace=True)

    # Drop all NaN
    df.dropna(inplace=True)

    # Change IDs to string value for streaming images
    df["spid"] = df["spid"].astype("string")

    # Keep just the 8 digit value in spid, removing float value
    df["spid"] = df["spid"].str.extract("(\d+).", expand=False)

    return df


def clean_hats_rounds(round_hat_data):
    """Processes the rounds data for all 10 rounds using variables
    that are important for NHATs dementia classification labeling
    """

    # Fill NaN values with arbitrary int value.
    round_hat_data["cp1dad8dem"] = round_hat_data["cp1dad8dem"].fillna(10)

    # Diagnosis variables
    hc_items = [
        ("2 NO", 2),
        (" 2 NO", 2),
        ("1 YES", 1),
        (" 1 YES", 1),
        ("-9 Missing", 10),
        ("-8 DK", 0),
        ("7 PREVIOUSLY REPORTED", 7),
        ("-1 Inapplicable", 10),
        ("-7 RF", 10),
    ]
    ad8dem = [
        ("1 DEMENTIA RESPONSE TO ANY AD8 ITEMS IN PRIOR ROUND", 1),
        ("1 DEMENTIA RESPONSE TO ANY AD8 ITEMS IN PRIOR ROUNDS", 1),
        ("-1 Inapplicable", 0),
    ]

    # Executive Functioning Clock Drawing item
    cg_items = [
        ("-2 Proxy says cannot ask SP", np.nan),
        ("-7 SP refused to draw clock", np.nan),
        ("-4 SP did not attempt to draw clock", np.nan),
        ("-3 Proxy says can ask SP but SP unable to answer", np.nan),
        ("-1 Inapplicable", np.nan),
        ("-9 Missing", np.nan),
        ("4 Reasonably accurate depiction of a clock", 4),
        ("3 Mildly distorted depiction of a clock", 3),
        ("2 Moderately distorted depection of a clock", 2),
        ("2 Moderately distorted depiction of a clock", 2),
        ("5 Accurate depiction of a clock (circular or square)", 5),
        ("1 Severely distorted depiction of a clock", 1),
        ("0 Not recognizable as a clock", 0),
    ]

    # Orientation Variables
    pres_first = [
        (" 1 Yes", 1),
        ("-1 Inapplicable", 0),
        (" 2 No", 0),
        ("-7 RF", 0),
        ("-9 Missing", np.nan),
        (" 1 YES", 1),
        (" 2 NO", 0),
    ]
    pres_last = [
        (" 1 Yes", 1),
        ("-1 Inapplicable", 0),
        (" 2 No", 0),
        ("-7 RF", 0),
        ("-9 Missing", np.nan),
        (" 1 YES", 1),
        (" 2 NO", 0),
    ]
    vp_first = [
        (" 1 Yes", 1),
        ("-1 Inapplicable", 0),
        (" 2 No", 0),
        ("-7 RF", 0),
        ("-9 Missing", np.nan),
        (" 1 YES", 1),
        (" 2 NO", 0),
    ]
    vp_last = [
        (" 1 Yes", 1),
        ("-1 Inapplicable", 0),
        (" 2 No", 0),
        ("-7 RF", 0),
        ("-9 Missing", np.nan),
        (" 1 YES", 1),
        (" 2 NO", 0),
    ]
    ans_yr = [
        ("1 YES", 1),
        ("2 NO/DON'T KNOW", 0),
        ("-1 Inapplicable", np.nan),
        ("-9 Missing", np.nan),
        (" 1 YES", 1),
        (" 2 NO", 0),
        (" 2 NO/DON'T KNOW", 0),
        ("-7 RF", 0),
    ]
    ans_day = [
        ("1 YES", 1),
        ("2 NO/DON'T KNOW", 0),
        ("-1 Inapplicable", np.nan),
        ("-9 Missing", np.nan),
        (" 1 YES", 1),
        (" 2 NO", 0),
        (" 2 NO/DON'T KNOW", 0),
        ("-7 RF", 0),
    ]
    ans_month = [
        ("1 YES", 1),
        ("2 NO/DON'T KNOW", 0),
        ("-1 Inapplicable", np.nan),
        ("-9 Missing", np.nan),
        (" 1 YES", 1),
        (" 2 NO", 0),
        (" 2 NO/DON'T KNOW", 0),
        ("-7 RF", 0),
    ]
    ans_dow = [
        ("1 YES", 1),
        ("2 NO/DON'T KNOW", 0),
        ("-1 Inapplicable", np.nan),
        ("-9 Missing", np.nan),
        (" 1 YES", 1),
        (" 2 NO", 0),
        (" 2 NO/DON'T KNOW", 0),
        ("-7 RF", 0),
    ]

    # Memory Variables
    delay_wrds = [
        ("-3 Proxy says can ask SP but SP unable to answer", 0),
        ("-2 Proxy says cannot ask SP", 0),
        ("-7 SP refused activity", 0),
        ("-1 Inapplicable", np.nan),
        ("-9 Missing", np.nan),
    ]
    immed_wrds = [
        ("-3 Proxy says can ask SP but SP unable to answer", 0),
        ("-2 Proxy says cannot ask SP", 0),
        ("-7 SP refused activity", 0),
        ("-1 Inapplicable", np.nan),
        ("-9 Missing", np.nan),
    ]

    # Diagnosis Items
    for item in hc_items:
        round_hat_data.hc1disescn9.replace(item[0], item[1], inplace=True)
    for item in ad8dem:
        round_hat_data.cp1dad8dem.replace(item[0], item[1], inplace=True)

    # Executive functioning, clock drawing
    for item in cg_items:
        round_hat_data.cg1dclkdraw.replace(item[0], item[1], inplace=True)
    # Drop all NaN
    round_hat_data.dropna(inplace=True)

    # Orientation, Pres Last/First name, VP Last/First, Month, Day, Year, DOW
    for item in pres_first:
        round_hat_data.cg1presidna3.replace(item[0], item[1], inplace=True)
    for item in pres_last:
        round_hat_data.cg1presidna1.replace(item[0], item[1], inplace=True)
    for item in vp_first:
        round_hat_data.cg1vpname3.replace(item[0], item[1], inplace=True)
    for item in vp_last:
        round_hat_data.cg1vpname1.replace(item[0], item[1], inplace=True)
    for item in ans_month:
        round_hat_data.cg1todaydat1.replace(item[0], item[1], inplace=True)
    for item in ans_day:
        round_hat_data.cg1todaydat2.replace(item[0], item[1], inplace=True)
    for item in ans_yr:
        round_hat_data.cg1todaydat3.replace(item[0], item[1], inplace=True)
    for item in ans_dow:
        round_hat_data.cg1todaydat4.replace(item[0], item[1], inplace=True)

    # Memory Items, Delay word recall, Immediate Word recall
    for item in delay_wrds:
        round_hat_data.cg1dwrdimmrc.replace(item[0], item[1], inplace=True)
    for item in immed_wrds:
        round_hat_data.cg1dwrddlyrc.replace(item[0], item[1], inplace=True)

    # Change IDs to string value for streaming images
    round_hat_data["spid"] = round_hat_data["spid"].astype("string")

    # Keep just the 8 digit value in spid, removing float value
    round_hat_data["spid"] = round_hat_data["spid"].str.extract("(\d+).", expand=False)

    return round_hat_data
