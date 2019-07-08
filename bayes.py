CLASS_1 = 1
CLASS_2 = 2
# Returns a list of labels
# NOTE: Assumes a priori probabilities of each class are equal to 1/2.
def bayesian_classify(px_1,px_2):
    label_test = []
    for p1, p2 in zip(px_1, px_2):
        if p1 > p2:
            label_test.append(CLASS_1)
        else:
            label_test.append(CLASS_2)

    return label_test