def rand(labels_1, labels_2):

    a = 0
    b = 0
    c = 0
    d = 0

    for i in range(labels_1.shape[0]):
        for j in range(i+1, labels_2.shape[0]):
            if labels_1[i] == labels_1[j] and labels_2[i] == labels_2[j]:
                a += 1
            if labels_1[i] == labels_1[j] and labels_2[i] != labels_2[j]:
                b += 1
            if labels_1[i] != labels_1[j] and labels_2[i] == labels_2[j]:
                c += 1
            if labels_1[i] != labels_1[j] and labels_2[i] != labels_2[j]:
                d += 1

    return (a+d) / (a+b+c+d)