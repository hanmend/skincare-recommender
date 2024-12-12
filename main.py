import pandas as pd
import glob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from products import Products

# This script focuses on Skincare, but there are other product categories in the dataset you could explore
product_category = "Skincare"
skin_type_input = input("What is your skin type? (Normal, Dry, Oily, Combination)? ")
skin_type = (
    None
    if skin_type_input.lower() not in ["normal", "dry", "oily", "combination"]
    else skin_type_input
)
budget = input("Do you have a budget? Y/N ")
assert budget.lower() in ["y", "n"]
if budget.lower() == "y":
    price_limit = input(
        "What is your price limit in USD? Please enter a whole number: "
    )
    assert price_limit.isnumeric()
    price_limit = int(price_limit)
else:
    price_limit = None

print("Generating product options...")

products = Products(product_category, price_limit)
product_options = products.get_products()
brand_interest = input("Is there a certain brand you are interested in? Y/N ")
assert brand_interest.lower() in ["y", "n"]
if brand_interest.lower() == "y":
    brand = input("What brand are you interested in? ")
    assert brand.lower() in [
        x.lower() for x in product_options["brand_name"].unique()
    ], f"Please select a brand name from the following list:\n {product_options['brand_name'].unique()}"
    products.set_brand(brand)
    product_options = products.get_products()
else:
    brand = None

print("Accessing product reviews...")

reviews_files = glob.glob("data/reviews*.csv")
df_list = []
for file in reviews_files:
    df = pd.read_csv(file, low_memory=False)
    df_list.append(df)
reviews_complete = pd.concat(df_list, ignore_index=True, sort=False)
product_ids = product_options["product_id"]
reviews = reviews_complete[reviews_complete["product_id"].isin(product_ids)]

query = input("What type of product are you looking for? Enter your query: ")

print("Compiling product reviews by product...")

review_text_only_df = reviews[["product_id", "review_text"]].fillna("")
merged_reviews = review_text_only_df.groupby('product_id')['review_text'].agg(', '.join).reset_index()

print("Analyzing reviews and determining product relevance to your query...")


def compute_cosine_similarity(queries, df, column_name):
    vectorizer = TfidfVectorizer()
    text_vectors = vectorizer.fit_transform(df[column_name].fillna(""))
    query_vectors = vectorizer.transform(queries)
    similarity_scores = cosine_similarity(query_vectors, text_vectors)
    result_df = pd.DataFrame(similarity_scores, columns=df["product_id"])

    return result_df


similarity_df = compute_cosine_similarity(
    queries=[query], df=merged_reviews, column_name="review_text"
)

print("Gathering recommendations...")


def get_top_n_products(df, n=5):
    top_n = []
    for i in range(n):
        max_column = df.idxmax(axis=1)
        top_n.append(max_column[0])
        df = df.drop(max_column[0], axis=1)
    return top_n


top_5_recs = get_top_n_products(similarity_df, n=5)

for i in range(len(top_5_recs)):
    print(f"\nRecommendation #{i+1}:")
    product_name, brand_name, price, rating = products.get_product_info(top_5_recs[i])
    print(
        f"**{product_name.upper()}**, from the brand ~{brand_name}~. RATING: {rating}, PRICE ${price:.2f}"
    )
