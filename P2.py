import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

nba = pd.read_csv('nba_stats.csv')

classColumn = 'Pos'
attributes = ['MP','FGA', '3PA', '2PA', 'FTA', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS']

nbaAttributes = nba[attributes]
nbaClass = nba[classColumn]

# 80% of the data for training and the rest for validation
# Train_size=0.80, test_size=0.20
train_feature, test_feature, train_class, test_class = train_test_split(nbaAttributes, nbaClass, stratify=nbaClass, test_size=0.20, random_state=0)

# Naive Bayes classifier
nb = GaussianNB().fit(train_feature, train_class)
print("[GaussianNB Classifier] Test set score: {:.3f}".format(nb.score(test_feature, test_class)))

# Print out the training and validation set accuracy of the model
print("\nTraining set score: {:.3f}".format(nb.score(train_feature, train_class)))
print("Validation set score: {:.3f}".format(nb.score(test_feature, test_class)))

prediction = nb.predict(test_feature)
print("\nConfusion matrix:")
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

# Apply 10-fold stratified cross-validation 
scores = cross_val_score(nb, nbaAttributes, nbaClass, scoring='precision_macro', cv=10)
print("\nNAIVE BAYES 10-Fold Cross-validation scores: {}".format(scores))
print("\nNAIVE BAYES 10-Fold Average cross-validation score: {:.2f}".format(scores.mean()))
