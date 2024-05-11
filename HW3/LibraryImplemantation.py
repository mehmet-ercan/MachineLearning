import matplotlib.pyplot as plt
import warnings

from keras.datasets import mnist

warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

n_samples = 1000
n_components_list = [10, 50, 100, 200]
results = {}
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

for n_components in n_components_list:
    # Perform PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Plot class locations on 2D map
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(f'PCA with {n_components} components')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 2], c=y_test)
    plt.xlabel('First Principal Component')
    plt.ylabel('Third Principal Component')
    plt.title(f'PCA with {n_components} components')
    plt.show()

    # Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_pca, y_train)

    # Evaluate using cross-validation
    scores = cross_val_score(clf, X_train_pca, y_train, cv=5)
    results[n_components] = scores.mean()

# Print classification results
print("Classification Results:")
for n_components, score in results.items():
    print(f"PCA with {n_components} components: {score:.3f}")