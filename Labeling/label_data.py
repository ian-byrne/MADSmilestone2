"""Functions to create labels for the round data and to extract IDs with labels per
clock drawing round and save as a dictionary."""


def create_labels(rounds_clean):
    """Create labels: label 0 = pre-dementia, 1 = post-dementia, 2 = normal
    Create Normal Label for all entries, base case, then assign value
    of 1 to 1's and 7's"""

    df = rounds_clean

    # Assign label of 2 or 1
    df['label'] = [2 if x == 2 else 1 for x in df['hc1disescn9']]

    # Changing Round 1-3
    for rounds in df['round'].unique()[:-1]:
        round_val = rounds

        round_post = df[df['round'] == (round_val + 1)]
        # assign label value of 0 to the previous round where diagnosis changed from a 2.0 to 1.0
        # diagnoses value of 7 corresponds to previous round value of 1, label stays a 1

        post1 = round_post[round_post['hc1disescn9'] == 1]
        df.loc[(df['round'] == round_val) & df['spid'].isin(post1['spid']), 'label'] = 0

    return df


def get_ids(df):
    """ Creates dictionary of rounds keys with values as list of tuples
     containing SPIDs and labels per round"""

    # id_dict = {}
    # for val in df['round'].unique():
    # id_dict[val] = df[df['round'] == val]['spid'].values

    indexed_df = df[['round', 'spid', 'label']]
    d = {}
    #for round, id, label in zip(indexed_df['round'].values,
    #                            indexed_df['spid'].values, indexed_df['label'].values):
    #    d.setdefault(round, {}).update({id: label})
    for round, id, label in zip(indexed_df['round'].values, indexed_df['spid'].values, indexed_df['label'].values):
        d.setdefault(round, []).append((id, label))

    return d





def create_hats_labels(round_hat_clean):
    # Create Labels
    ## PROBABLE DEMENTIA: 1
    # First check if Dx is 1 - Yes, 2 - No, 7 - Previously diagnosed
    # Then if no diagnosis, check proxy input for diagnosis, 1 - YES or 0 for inapplicable
    # If inapplicable, then check to see if 2 domains meet cut off

    ## POSSIBLE DEMENTIA: 0
    # Check if 1 domain meets cut off

    ## NO DEMENTIA: 2 (All else)

    # Create a list of conditions
    conditions = [
        (round_hat_clean['hc1disescn9'] == 1) | (round_hat_clean['hc1disescn9'] == 7) | (
                    round_hat_clean['hc1disescn9'] == 0),
        (round_hat_clean['cp1dad8dem'] == 1),
        (round_hat_clean['orientation_score'] <= 3) & (round_data['memory_score'] <= 3.0),
        (round_hat_clean['orientation_score'] <= 3) & (round_data['cg1dclkdraw'] <= 1),
        (round_hat_clean['memory_score'] <= 3.0) & (round_hat_clean['cg1dclkdraw'] <= 1),
        (round_hat_clean['memory_score'] > 3.0) & (round_hat_clean['cg1dclkdraw'] > 1) & (
                    round_hat_clean['orientation_score'] <= 3),
        (round_hat_clean['memory_score'] > 3.0) & (round_hat_clean['cg1dclkdraw'] <= 1) & (
                    round_hat_clean['orientation_score'] > 3),
        (round_hat_clean['memory_score'] <= 3.0) & (round_hat_clean['cg1dclkdraw'] > 1) & (
                    round_hat_clean['orientation_score'] > 3),
        (round_hat_clean['memory_score'] > 3.0) & (round_hat_clean['cg1dclkdraw'] > 1) & (
                    round_hat_clean['orientation_score'] > 3),
        (round_hat_clean['memory_score'] > 3.0) & (round_hat_clean['cg1dclkdraw'] > 1) & (
                    round_hat_clean['orientation_score'] > 3),
        (round_hat_clean['memory_score'] > 3.0) & (round_hat_clean['cg1dclkdraw'] > 1) & (
                    round_hat_clean['orientation_score'] > 3)]

    # Create a list of the values we want to assign for each condition
    values = [1, 1, 1, 1, 1, 0, 0, 0, 2, 2, 2]

    # Add to the column and use np.select to assign values to it using lists as arguments
    round_hat_clean['label'] = np.select(conditions, values)

    return round_hat_clean
