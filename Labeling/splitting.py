from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def split_data(df, first_split, second_split, label):
    """
    :param df: pass in the dataframe
    :param first_split: how much you want to split the test/val set from the train set, (percent
    of test/val set)
    :param second_split: percent to split the test and validation set (typical is 0.5)
    :return: returns the training set dataframe, validation set dataframe, and test set dataframe
    """
    data = df.copy()

    train, temp = train_test_split(data, test_size=first_split, random_state=42)
    test, val = train_test_split(temp, test_size=second_split, random_state=42)

    print("length of training set: ", len(train))
    print("length of validation set: ", len(val))
    print("length of test set: ", len(test))
    print("Total length: ", len(train) + len(val) + len(test))

    print("Train data: \n")
    plt.hist(train[label])
    plt.show()
    print("Val data: \n")
    plt.hist(val[label])
    plt.show()
    print("Test data: \n")
    plt.hist(test[label])
    plt.show()

    return train, val, test
