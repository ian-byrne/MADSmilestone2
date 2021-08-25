def clean_data(rounds_df):
    # Clean the row values in hc1disescn9 , cg1dclkdraw
    # hc1disescn9 asks if subject has dementia or alzheimers: 1 YES, 2 NO

    df = rounds_df
    hc_items = [('2 NO', 2), (' 2 NO', 2), ('1 YES', 1), (' 1 YES', 1),
                ('-9 Missing', np.nan), ('-8 DK', np.nan), ('7 PREVIOUSLY REPORTED', 7),
                ('-1 Inapplicable', np.nan), ('-7 RF', np.nan)]

    cg_items = [('-2 Proxy says cannot ask SP', np.nan), ('-7 SP refused to draw clock', np.nan),
                ('-4 SP did not attempt to draw clock', np.nan),
                ('-3 Proxy says can ask SP but SP unable to answer', np.nan),
                ('-1 Inapplicable', np.nan), ('-9 Missing', np.nan)]

    for item in hc_items:
        df.hc1disescn9.replace(item[0], item[1], inplace=True)

    # Remove cg1dclkdraw subjects that did not draw a clock, or image data is missing
    for item in cg_items:
        df.cg1dclkdraw.replace(item[0], item[1], inplace=True)

    # Drop all NaN
    df.dropna(inplace=True)

    # Change IDs to string value for streaming images
    df['spid'] = df['spid'].astype('string')

    # Keep just the 8 digit value in spid, removing float value
    df['spid'] = df['spid'].str.extract('(\d+).', expand=False)

    return df