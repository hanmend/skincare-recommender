import pandas as pd

class Products():
    def __init__(self, category, price_limit=None):
        self.complete_dataset = pd.read_csv("archive/product_info.csv")
        self.category = category
        self.price_limit = price_limit
        self.brand = None

    def get_products(self):
        df = self.complete_dataset[self.complete_dataset['primary_category'] == self.category]
        if self.brand != None: 
            df = df[df['brand_name'] == self.brand]
        if self.price_limit != None: 
            df = df[df['price_usd'] <= self.price_limit]

        return df
    
    def set_brand(self, brand_name):
        self.brand = brand_name

    def get_product_info(self, product_id):
        products = self.get_products()
        product = products[products['product_id'] == product_id]

        product_name = product['product_name'][0]
        brand_name = product['brand_name'][0]
        price = product['price_usd'][0]
        rating = product['rating'][0]

        return product_name, brand_name, price, rating
