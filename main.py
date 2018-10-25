from CNN import CNN
from LSTM import LSTM
import json, pickle

data_folder = '/home/aida/Projects/IE-hw2/data/'


def main():
    params = json.load(open("params.json", "r"))
    try:
        data = pickle.load(open("data.pkl", "rb"))
    except Exception:
        print("Error: run make_vocab.py first")
        exit(1)

    if params["pretrain"] and data["embeddings"] == []:
        print("No embeddings found, please run make_vocab.py")
        exit(1)

    if params["model"] == "CNN":
        cnn = CNN(params, data["vocab"], data["embeddings"])
        cnn.build()
        cnn.run_model(data["train_batches"], data["dev_batches"], data["true_dev_labels"], data["tag_dict"])
    else:
        lstm = LSTM(params, data["vocab"], data["embeddings"])
        lstm.build()
        predictions = lstm.run_model(data)
    with open(data_folder + "test_output.txt", "w") as out:
        for i, pred in enumerate(predictions):
            out.write(str(8000 + i) + "\t" + pred + "\n")



if __name__ == '__main__':
    main()