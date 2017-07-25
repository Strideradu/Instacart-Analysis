
# Use python 3
# alias python=python3

# Make folders
# mkdir -p features
# mkdir -p data

# Define training, validation and test set
python train_val_test_customers.py
python userXproduct.py

# Generate features
python product_features.py
python user_features.py
python order_features.py
python userXproduct_features.py
