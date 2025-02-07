from run import *
from sklearn import metrics
from sklearn.metrics import f1_score

# import matplotlib.pyplot as plt

from tqdm import tqdm


if __name__ == "__main__":
    # dataset
    rows = read_data("./fever_test.jsonl", to_dataset=False)

    MODEL = "./logs"
    # init model
    model_module = BERT(MODEL, num_labels=3)

    pred = []; gold = []
    for row in tqdm(rows, total=len(rows)):
        try:
            # run instance
            pred.append(model_module.run(row['text'])); gold.append(row['label'])
        except:
            continue
    
    accuracy = np.sum(np.array(pred) == np.array(gold)) / len(gold)
    print("ACCURACY:", accuracy)
    # f1 score
    for mode in ['macro', 'weighted']:
        f1 = f1_score(gold, pred, average=mode)
        print(f"{mode.upper()} F1: {f1}")
    print('-------------------------')

    # confusion matrix
    print(metrics.confusion_matrix(gold, pred))
