
import matplotlib.pyplot as plt
import seaborn as sns


def watch_data_info(data):

    # This function returns the first 5 rows for the object based on position.
    # It is useful for quickly testing if your object has the right type of data in it.
    print(data.head())

    # This method prints information about a DataFrame including the index dtype and column dtypes,
    # non-null values and memory usage.
    print(data.info())

    # Descriptive statistics include those that summarize the central tendency,
    # dispersion and shape of a dataset’s distribution, excluding NaN values.
    print(data.describe(include='all').transpose())


def print_data(data):

    unique_users = data['UserId'].nunique()
    print(f"number of users are :  {unique_users}")
    unique_products = data['ProductId'].nunique()
    print(f"number of products ranked are : {unique_products}")
    # number of ranking is the same amount as number of ratings
    print(f"number of ranking are: {len(data['Rating'])}")
    # get number of ranking for each product in file
    rankings_per_product = data['ProductId'].value_counts()
    # now extract min and max from that
    print(f"minimum number of ratings given to a product : {rankings_per_product.min()}")
    print(f"maximum number of ratings given to a product : {rankings_per_product.max()}")
    # get number of ranking for each user in file
    rankings_per_user = data['UserId'].value_counts()
    # now extract min and max from that
    print(f"minimum number of products ratings by user : {rankings_per_user.min()}")
    print(f"maximum number of products ratings by user : {rankings_per_user.max()}")

