with open('eggs2.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['Id',  'Category'])
        for i in range(pred_labels_test.shape[0]):
            spamwriter.writerow([str(i),  str(pred_labels_test[i])]