"""Prepares data for Lab 2

This script adds information to data/ratings.csv for Lab 2 based on concepts
learned during Lab 1. Students aren't expected to go through this, but it's
here for the curious.
"""

import cupy as cp
import cupyx.scipy.sparse
import cudf
import numpy as np

ratings = cudf.io.csv.read_csv("data/raw_data.csv")

# Split data into train and valid
count_ratings = len(ratings)
scaled_index = ratings.index / count_ratings
ratings["timeBreaker"] = scaled_index + ratings["unixReviewTime"]

valid_ratings = (
    ratings[["reviewerID", "timeBreaker"]].groupby(["reviewerID"], as_index=False).max()
)
valid_ratings["valid"] = True
valid_ratings.head()

data_split = ratings.merge(valid_ratings, how="left", on=["reviewerID", "timeBreaker"])
data_split["valid"] = ~data_split["valid"].isnull()

# Give user, item, and brand unique integer indexes
user_indexes, user_mapping = data_split["reviewerID"].factorize()
item_indexes, item_mapping = data_split["asin"].factorize()
user_count = user_mapping.count()
item_count = item_mapping.count()

data_split["user_index"] = user_indexes
data_split["item_index"] = item_indexes
data_split["brand_index"] = data_split["brand"].factorize()[0]
print("Data split.")

# Train ALS
train_indexes = ~data_split["valid"]
valid_indexes = data_split["valid"]
shape = (user_count, item_count)
overall = data_split["overall"]


def get_dataset(data_selector, user_indexes, item_indexes, overall, shape):
    row = cp.asarray(user_indexes[data_selector])
    column = cp.asarray(item_indexes[data_selector])
    data = cp.asarray(overall[data_selector])

    sparse_data = cupyx.scipy.sparse.coo_matrix((data, (row, column)), shape=shape)
    mask = cp.asarray([1 for _ in range(len(data))], dtype=np.float32)
    sparse_mask = cupyx.scipy.sparse.coo_matrix((mask, (row, column)), shape=shape)
    return row, column, data, sparse_data, sparse_mask


train_row, train_column, train_data, train_sparse, train_mask = get_dataset(
    train_indexes, user_indexes, item_indexes, overall, shape
)

valid_row, valid_column, valid_data, valid_sparse, valid_mask = get_dataset(
    valid_indexes, user_indexes, item_indexes, overall, shape
)


def initalize_features(length, embeddings=2):
    return cp.random.rand(embeddings, length) * 2 - 1


user_features = initalize_features(shape[0])
item_features = initalize_features(shape[1])


def rmse(user_features, item_features, data, row, column):
    predictions = (user_features[:, row] * item_features[:, column]).sum(axis=0)
    mean_squared_error = ((predictions - data) ** 2).mean() ** 0.5
    return predictions, mean_squared_error


def als(values, mask, features, scale=0.01):
    numerator = values.dot(features.T)
    squared_features = (features ** 2).sum(axis=0)
    denominator = scale + mask.dot(squared_features)
    return (numerator / denominator[:, None]).T


# For this dataset, converges when rmse is less than 1.21 on valid set
error = 10
while error > 1.21:
    user_features = als(train_sparse, train_mask, item_features)
    item_features = als(train_sparse.T, train_mask.T, user_features)
    predictions, error = rmse(
        user_features, item_features, valid_data, valid_row, valid_column
    )
    print ("Valid RMSE:", error)
row = cp.asarray(user_indexes)
column = cp.asarray(item_indexes)
data = cp.asarray(overall)

print("ALS trained.")

# Add predictions to dataset
predictions = (user_features[:, row] * item_features[:, column]).sum(axis=0)
data_split["als_prediction"] = predictions
data_split["user_embed_0"] = user_features[0, row]
data_split["user_embed_1"] = user_features[1, row]
data_split["item_embed_0"] = item_features[0, row]
data_split["item_embed_1"] = item_features[1, row]

# Add category codes
category_columns = ["category_1_2"]
categories = data_split["category_0_2"]

for category in category_columns:
    categories = cudf.concat([categories, data_split[category]])
# Originally removed "Electronics" and "NA" but including them is easier for
# Neural net to digest
categories = categories.unique()
categories = cudf.DataFrame({"category": categories})
categories["index"] = categories.index

merged_categories = data_split.merge(
    categories, how="left", left_on="category_0_2", right_on="category"
)
merged_categories = merged_categories.drop("category")
merged_categories = merged_categories.rename(columns={"index": "category_0_2_index"})

merged_categories = merged_categories.merge(
    categories, how="left", left_on="category_1_2", right_on="category"
)
merged_categories = merged_categories.drop("category")
merged_categories = merged_categories.rename(columns={"index": "category_1_2_index"})

cudf.io.csv.to_csv(merged_categories, "data/task_2.csv", index=False)
print("Categories merged.")

# For assessment, need to separate ratings and metadata
# This is kinda a cheat that should be done with the metadata before joining,
# but since each item has a review, this can be done.

save_columns = ["user_index", "item_index", "overall", "valid"]
cudf.io.csv.to_csv(merged_categories[save_columns], "data/task_3_ratings.csv", index=False)

save_columns = [
    "item_index",
    "brand_index",
    "category_0_0",
    "category_0_1",
    "category_0_2",
    "category_0_3",
    "category_1_0",
    "category_1_1",
    "category_1_2",
    "category_1_3",
    "category_0_2_index",
    "category_1_2_index",
    "salesRank_Electronics",
    "salesRank_Camera",
    "salesRank_Computers",
    "salesRank_CellPhones",
    "salesRank_CellPhones_NA",
    "salesRank_Electronics_NA",
    "salesRank_Camera_NA",
    "salesRank_Computers_NA",
    "price_filled",
]

# Need average of price_filled. Explained in lab 3
merged_categories = (
    merged_categories[save_columns].groupby(save_columns[:-1], as_index=False).mean()
)
cudf.io.csv.to_csv(merged_categories[save_columns], "data/task_3_metadata.csv", index=False)
print("Assessment data generated.")