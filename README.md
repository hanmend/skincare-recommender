# skincare-recommender
CS410 at UIUC course project

## Purpose 

This software is created to recommend skincare products based on a user query about what they're looking for. We use a dataset from kaggle called 'Sephora Products and Skincare Reviews' which was scraped from Sephora's website with a python scraper in March 2023. The data contains metadata on the products, as well as another dataset with user reviews of the products. This is a very simple recommender which utilizes cosine similarity of the vectorized queries and product reviews, which are also weighted with TF-IDF in order to discount term frequency when there are a lot of text reviews for one product and not many for another.

## How to use 

To run this software, you will need to install scikit-learn. I used anaconda to manage my environment. You can create an environment with python=3.10 in order to be compatible with scikit-learn. If conda install scikit-learn still gives you an error for the import statements, try pip install scikit-learn instead.

You can take a look at the raw data if you would like in the "archive" folder of this repo. For more details and csv specs, visit the page on kaggle: [Sephora Products and Skincare Reviews on Kaggle](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews/data)

### Running the program 

You only need to run main.py. You will be prompted in the console to answer a few questions regarding the type of product you are looking for, including your skin type, your price range, and if you have a particular brand in mind. 

It will take several minutes for the "Compiling product reviews by product..." step, this is normal. After this step, the similarity scores will be computed, and then you will receive five product recommendations. Within those recommendations, you will receive: 
* product name
* brand
* star rating
* price

## Implementation 

The script uses the user prompts in order to filter down the dataset of possible products. Then, we fetch all the reviews corresponding to these valid product ids. 

In order to construct a text object with which to calculate cosine similarity, we then concatenate all the reviews for each product into one document. This is helpful for examining the similarity to all reviews together, since we can get a better idea of relevance with a higher TF. However, we do discount with TF-IDF weighting by using the TfidfVectorizer() class. This way, when there are simply more reviews for one product compared to another, we are still able to compare the products to one another even with varying amounts of reviews, so the similarity score will still be meaningful. 

After this, we have one similarity score for each product. To make the recommendations, we use idxmax() to find the top 5 similarity scores, and then we can output the recommendations as described above.
