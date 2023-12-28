
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset

# load dataset
dataset = load_dataset("imdb")

# Extract training/testing text & labels
train_texts = dataset['train']['text']
train_labels = dataset['train']['label']

test_texts = dataset['test']['text']
test_labels = dataset['test']['label']

# Printing data
# for i in range(3):
#     print(f"Training Sample {i + 1}:")
#     print(f"Text: {train_texts[i]}")
#     print(f"Label: {train_labels[i]}")
#     print("\n")
#
#     print(f"Test Sample {i + 1}:")
#     print(f"Text: {test_texts[i]}")
#     print(f"Label: {test_labels[i]}")
#     print("\n")

# Max = 100,000 features (words)
vect = CountVectorizer(max_features=100000)

# fit on training data & transform into bag-of-words
# bow = 'bag-of-words'
train_bow = vect.fit_transform(train_texts)
# test_bow = vect.transform(test_texts)  #

# Added scaler to try to fix accuracy
scaler = StandardScaler(with_mean=False)

train_bow_scaled = scaler.fit_transform(train_bow)
test_bow_scaled = scaler.transform(vect.transform(test_texts))

# Printing the bag-of-words representation of the training data
# print(train_bow)

model = LogisticRegression(max_iter=1000)  # Increased max_iter to fix accuracy (Ended up lowering Acc)
model.fit(train_bow_scaled, train_labels)

predictions = model.predict(test_bow_scaled)

accuracy = (predictions == test_labels).mean()
print(f"Test Accuracy: {accuracy}")


# Test 1 - LR (LBFGS) prob reached max number of iterations without achieving convergence. - Accuracy: 0.86656 (86.656%)
# Test 2 - Increased max_iter to 1000. - Accuracy: 0.85372 (85.372%)
# Test 3 - Added 'StandardScaler' to scale data. - Accuracy: 0.82252 (82.252%)
# Test 4 - Removed stop words using "stop_words='english'", Ended up lowering accuracy. - Accuracy: 0.8096 (80.96%)
# Test 5 - Increasing max_features from 10K to 100K increased accuracy. - Accuracy: 0.8326 (83.26%)

# Highest Accuracy: Test 1
