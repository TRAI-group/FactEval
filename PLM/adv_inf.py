import csv
from run import *
from sklearn import metrics
from sklearn.metrics import f1_score

# import matplotlib.pyplot as plt

from tqdm import tqdm


label_dict1 = {'SUPPORTED': 0, 'REFUTED': 1, 'NEI': 2}
# label_dict1 = {'0': 'SUPPORTED', '1': 'REFUTED', '2': 'NEI'}

def read_adv_data(path):
    '''
    read adv csv
    '''
    dataset = {}
    cnt=0
    with open(path, mode ='r', encoding='latin1')as f:
        data_f = csv.reader(f)
        # skip header
        next(data_f)

        for line in data_f:
            cnt=cnt+1
           
            # label = label_dict1[line[-1]]
            label = line[-1]

            dataset[str(line[0])] = {
                'text': f"{line[3].strip()}[SEP]{line[2].strip()}",
                'label': label
            }
    return dataset

            
if __name__ == "__main__":
    # dataset
    rows = read_data("./fever_test.jsonl", to_dataset=False)
    # load adv dataset
    adv_rows = read_adv_data("data/adv_data/leet_25_adv_test_set.csv")
    # print(len(adv_rows))
    # print("len", len(rows), len(adv_rows))

    MODEL = "./logs"
    # init model
    model_module = BERT(MODEL, num_labels=3)

    adv_check = 0; adv_success = 0

    pred = []; adv_pred = []; gold = []; cnt=0
    for row in tqdm(rows, total=len(rows)):
        print("text",row['text'], adv_rows[str(row['id'])]['text'])
        cnt = cnt+1
        # if(cnt>5):
        #     break
        try:
            predicted = model_module.run(row['text']); label = row['label']
            # check
            if predicted == label:
                adv_check += 1
                # rerun with adv claim
                adv_predicted = model_module.run(adv_rows[str(row['id'])]['text'])
                # print("adv_predicted", adv_predicted, predicted)
                if adv_predicted != predicted:
                    # on success
                    adv_success += 1
            else:
                adv_predicted = predicted
        except:
            continue

        pred.append(predicted); adv_pred.append(adv_predicted); gold.append(label)

    for y_pred, y_gold in zip([pred, adv_pred], [gold, gold]):
        # for a pair
        accuracy = np.sum(np.array(y_pred) == np.array(y_gold)) / len(y_gold)
        print("ACCURACY:", accuracy)
        # f1 score
        for mode in ['macro', 'weighted']:
            f1 = f1_score(y_gold, y_pred, average=mode)
            print(f"{mode.upper()} F1: {f1}")
        print('-------------------------\n')

        # confusion matrix
        print(metrics.confusion_matrix(y_gold, y_pred), '\n')
    # print("**********************",len(pre))

    print(f"adv check: {adv_check}, adv success: {adv_success}")
    ##
    print(f"adv success rate: {(adv_success / adv_check) * 100}")
