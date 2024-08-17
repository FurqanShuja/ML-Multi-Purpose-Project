import random
import matplotlib.pyplot as plt
import pandas as pd

#Getting our data into dataframes
FeaturesTrain = pd.read_csv("features-train.tsv", sep = "\t")
FeaturesTest = pd.read_csv("features-test.tsv", sep = "\t")
ClarityScores = pd.read_csv("clarity-scores-train.tsv", sep = "\t")
QualityScores = pd.read_csv("quality-scores-train.tsv", sep = "\t")

# Plot scatterplot for the selected feature and clarity class
def scatter_plot(feature, clarity):
    plt.scatter(feature, clarity) 
    plt.xlabel("Selected Feature")  
    plt.ylabel("Clarity Class")  
    plt.title("Feature vs Clarity Class")
    plt.show() 

# Plot the line of best fit
def line_plot(feature, clarity, weights):
    plt.scatter(feature, clarity)
    plt.plot(feature, [weights[0] + weights[1]*x for x in feature], color='red')
    plt.xlabel("Selected Feature")
    plt.ylabel("Clarity Class")
    plt.title("Line of Best Fit")
    plt.show()

# Construct LMS algorithm and find the weight vector
def lms_algorithm(features, clarity):

    w0, w1 = 0, 0 
    alpha = 0.01  
    num_iterations = 1000 

    for iteration in range(num_iterations):
        for i in range(len(features)):

            predicted_value = w0 + w1 * features[i]

            error = clarity[i] - predicted_value

            w0 = w0 + alpha * error 

            w1 = w1 + alpha * error * features[i]

    return w0,w1  

# Compute the Residual Sum of Squares (RSS)
def find_rss(features, clarity, weights):

    rss = 0

    for x, y_actual in zip(features, clarity):

        y_predicted = weights[0] + weights[1] * x

        residual = y_actual - y_predicted

        rss += residual**2

    return rss

# Classify examples in the test set using the weight vector
def classify_test_set(features_test, weights):

    predicted_classes = [round(weights[0] + weights[1] * x) for x in features_test]

    return predicted_classes

# (a) Plot scatterplot
scatter_plot(FeaturesTrain ["misspelled_word_ratio"], ClarityScores ["clarity"])

# (b) Implement LMS algorithm
weights = lms_algorithm(FeaturesTrain ["misspelled_word_ratio"], ClarityScores ["clarity"])

# Plot the line of best fit
line_plot(FeaturesTrain ["misspelled_word_ratio"], ClarityScores ["clarity"], weights)

# (c) Compute RSS
rss = find_rss(FeaturesTrain ["misspelled_word_ratio"], ClarityScores ["clarity"], weights)
print(f"RSS: {rss}")

# (d) Classify test set and write to clarity-scores-test.tsv
predicted_classes_test = classify_test_set(FeaturesTest ["misspelled_word_ratio"], weights)
clarity_scores_test = pd.DataFrame({'#id': FeaturesTest['#id'], 'clarity': predicted_classes_test})
clarity_scores_test.to_csv("clarity_scores_test.tsv",sep='\t', index=False)


