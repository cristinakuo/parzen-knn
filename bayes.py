def bayesian_classify(px_1,px_2):
    label_test = []
    for p1, p2 in zip(px_1, px_2):
        if p1 > p2:
            label_test.append(1)
        else:
            label_test.append(2)

    return label_test