#%%
import hashlib as h
import numpy as np
import pandas as pd
import seaborn as sns

def return_hash_int(key):
    hashvalue = h.md5(str(key).encode()).hexdigest() #encodes variable key into bytes to be inputted into hash function, applies md5 algorithm and then makes outputs readable by using hexdigest()
    hash_int = int(hashvalue, 16)

    return hash_int


#%%
# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def create_sample_split(df, id_column, training_frac):
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.9

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """
    df["unique_identifier"] = df[id_column].apply(return_hash_int)
    
    b = 10**8
    
    # Take modulus to make the number smaller but preserve the randomness
    df["unique_identifier"] = df["unique_identifier"].apply(lambda x: x%b)

    #as we have the property that 0<=(a%b)<b we can normalise the values, this will ensure a uniform distribution
    df["unique_identifier"] = df["unique_identifier"].apply(lambda x: x/b)

    train_df = df[df["unique_identifier"]<=training_frac]
    test_df = df[df["unique_identifier"]>training_frac]




    return train_df,test_df

# %%
dta = sns.load_dataset("iris")

# Adding an id column that is == index
dta["id"] = dta.index

test = create_sample_split(dta, "id", training_frac=0.8)
print("Head of testing set: ")
print(test[1].head())
print("")
print(f"length of training set is {len(test[0])}")
print(f"length of testing set is {len(test[1])}")
print(f"Actual split train to test proportion: {  len(test[0])/   ( len(test[0])+  len(test[1]) )   }"   )
print("")

test_train_df = create_sample_split(dta, "id", training_frac=0.2)[0]
test_test_df = create_sample_split(dta, "id", training_frac=0.2)[1]

print(len(test_test_df) + len(test_test_df))
# %%
