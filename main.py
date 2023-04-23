import numpy as np
import os
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import graphviz
from sklearn.preprocessing import OneHotEncoder

"""
    IMPORTANT NOTES: PaRappa The Rapper on line 4222 original had the rating K-A. 
    On initial research, I'm not sure where this rating came from, thus I changed 
    it to rating E. this is the current ESRB rating of the game. This change may 
    be reverted. Unsure on correct path yet. Needs more research

    On line 5569, Supreme Ruler: Cold War is marked as having a RP rating, 
    meaning that the ESRB had yet to rate the game at the point of scraping the 
    data. It now has a rating of 'E10+' 
    (https://www.esrb.org/ratings/31253/supreme-ruler-cold-war/) For now, I have 
    modified the line to reflect that rating, but I'm not sure if I will keep this
    change or this game. 

    69 records don't have ratings. Nice. Not sure what to do about those. For now, I'm 
    just going to label them as -1 in rating conversion function
"""

"""
    RESULTS SO FAR: mean accuracy hovers around 60-70% when I have a tree with between 
    5 and 10 max depth. When the max depth is increased or uncaped, the accuracy is between 
    55-60%. Not sure what other variables I can modify with sklearn, but I plan
    to mess with other hyperparameters to better understand this model.

    Thus far, I have found that user count is the most important attribute to sales. Pretty 
    sure this could be the wrong cause/effect than I expected. May remove this attribute or 
    choose a different variable such as critic score as the target variable

    I also want to either add genre as a one-hot encoded attribute to this model or create a 
    different model that uses genre and see if the results differ at all. 

    model1: (decision tree) create a decision tree that tries to guess the 
    sales of a game. Modify a lot of aspects of this model to try to get it 
    to 75% accuracy

    model2: (decision tree) create a decision tree that tries to guess the 
    sales of a game, but add in the genre of game using one-hot encoding

    model3: (knn or linear regression) I want to plot the data and see if I can spot any trends 
    with the plotted elements. For example, maybe I could create a knn model that
    tries to predict the year. Or I could create a linear regression model that attempts
    to guess the year/price/etc.  
"""
pd.options.mode.chained_assignment = None  # default='warn'

RATING_CONVERSION_DICT = {"E": 0, "E10+": 1, "T": 2, "M": 3, "AO": 4}


# Taken from https://www.tensorflow.org/decision_forests/tutorials/beginner_colab
def split_dataset(dataset, test_ratio=0.20):
    """Splits a panda dataframe in two."""
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]


def categorize_column(df, column_name):
    df[column_name] = pd.Categorical(df[column_name])
    category_dict = dict(enumerate(df[column_name].cat.categories))
    df[column_name] = df[column_name].cat.codes
    return category_dict


def extract_target_column(df, column_name):
    return df[column_name], df.drop(column_name, axis=1)


# used for testing
def get_rows_without_ratings(df):
    return [j for i, j in df.iterrows() if j["Rating"] not in RATING_CONVERSION_DICT]


# read the file
df = pd.read_csv(
    os.path.join(os.getcwd(), "Cleaned Data 2.csv"),
    sep=r'(?!\B"[^"]*),(?![^"]*"\B)',
    engine="python",
)


def model_1(df):
    # select a few important
    # ["Name","Year_of_Release","Genre","Publisher","NA_Sales","EU_Sales","JP_Sales","Other_Sales","Global_Sales","Critic_Score","Critic_Count","User_Score","User_Count","Developer","Rating"]
    # FULL_LIST_OF_ATTRIBUTES = [
    #     "Name",
    #     "Year_of_Release",
    #     "Genre",
    #     "Publisher",
    #     "NA_Sales",
    #     "EU_Sales",
    #     "JP_Sales",
    #     "Other_Sales",
    #     "Global_Sales",
    #     "Critic_Score",
    #     "Critic_Count",
    #     "User_Score",
    #     "User_Count",
    #     "Developer",
    #     "Rating",
    # ]
    DATA_COLUMNS = ["User_Count", "User_Score", "Year_of_Release", "Critic_Count"]
    TARGET_COLUMN = "Global_Sales"
    ALL_COLUMNS = DATA_COLUMNS.copy()
    ALL_COLUMNS.extend([TARGET_COLUMN])
    df = df[ALL_COLUMNS]

    # turning user scores and global sales into discrete, integer values
    if "User_Score" in ALL_COLUMNS:
        df["User_Score"] = df["User_Score"].apply(lambda x: int(x * 10))

    df["Global_Sales"] = df["Global_Sales"].apply(lambda x: int(x * 100))

    # turning ratings into a numerical representation
    if "Rating" in ALL_COLUMNS:
        df["Rating"] = df["Rating"].apply(
            lambda x: RATING_CONVERSION_DICT[x] if x in RATING_CONVERSION_DICT else -1
        )

    bins = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 10000]
    labels = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 10000]

    # bin the sales
    df["Global_Sales"] = pd.cut(
        x=df["Global_Sales"], bins=bins, labels=labels, include_lowest=True
    )

    # split into training and test data
    train_ds, test_ds = split_dataset(df)

    train_y, train_x = extract_target_column(train_ds, "Global_Sales")
    test_y, test_x = extract_target_column(test_ds, "Global_Sales")

    # create the decision tree
    clf = tree.DecisionTreeClassifier(max_depth=4, max_leaf_nodes=20)
    clf = clf.fit(train_x, train_y)
    print(f"mean accuracy: {clf.score(test_x, test_y)}")

    # plot the tree
    plt.figure(figsize=(150, 18))
    tree.plot_tree(
        clf,
        fontsize=6,
        feature_names=DATA_COLUMNS,
        filled=True,
        class_names=[f"${str(x / 100)} million" for x in labels],
        max_depth=5,
    )  # proportion=True
    plt.savefig("model_1.png")
    # plt.show()


def model_2(df):
    # drop redundant sales numbers and some string attributes
    df = df.drop(["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"], axis=1)
    df = df.drop(["Name", "Publisher", "Developer"], axis=1)

    # turning user scores and global sales into discrete, integer values
    df["User_Score"] = df["User_Score"].apply(lambda x: int(x * 10))
    df["Global_Sales"] = df["Global_Sales"].apply(lambda x: int(x * 100))

    # # turning ratings into a numerical representation
    df["Rating"] = df["Rating"].apply(
        lambda x: RATING_CONVERSION_DICT[x] if x in RATING_CONVERSION_DICT else -1
    )

    # one hot encode genre
    ohe = OneHotEncoder()
    transformed = ohe.fit_transform(df[["Genre"]])
    df[ohe.categories_[0]] = transformed.toarray()
    df = df.drop(["Genre"], axis=1)

    bins = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 10000]
    # TODO Make more intuitive labels for when I plot the graph
    labels = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 10000]

    # bin the sales
    df["Global_Sales"] = pd.cut(
        x=df["Global_Sales"], bins=bins, labels=labels, include_lowest=True
    )

    # split into training and test data
    train_ds, test_ds = split_dataset(df)

    train_y, train_x = extract_target_column(train_ds, "Global_Sales")
    test_y, test_x = extract_target_column(test_ds, "Global_Sales")

    feature_names = [
        "Year_of_Release",
        "Critic_Score",
        "Critic_Count",
        "User_Score",
        "User_Count",
        "Rating",
    ].extend(
        transformed.toarray()
    )

    # create the decision tree
    clf = tree.DecisionTreeClassifier(
        max_depth=5,
    )
    clf = clf.fit(train_x, train_y)
    print(f"mean accuracy: {clf.score(test_x, test_y)}")

    # plot the tree
    plt.figure(figsize=(50, 18))
    tree.plot_tree(
        clf, fontsize=6, feature_names=feature_names, max_depth=5
    )  # class_names=[str(x) for x in labels], proportion=True
    plt.savefig("model_2.png")
    plt.show()


model_1(df)
# model_2(df)