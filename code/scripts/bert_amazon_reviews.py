"""
**********************
bert_amazon_reviews.py
**********************

CREDITS:
This code is based upon 
https://github.com/naveenjafer/BERT_Amazon_Reviews/blob/master/main.py

GOAL:
The purpose of this Python module is to demonstrate 
that BERT can be fine-tuned using the dataset of Amazon Reviews.

SPECIFICATIONS:
In this script, it will be made clear 
(1) how this dataset differs from that of Kaggle NER, and 
(2) how the resulting tensor differs from that of Kaggle NER.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import amazon_reviews

from transformers import  BertModel, BertTokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.is_available()

config = {
    "splitRatio" : 0.8,
    "maxLength" : 100,
    "printEvery" : 100,
    "outputFolder" : "Models",
    "outputFileName" : "AmazonReviewClassifier.dat",
    "threads" : 4,
    "batchSize" : 64,
    "validationFraction" : 0.0005,
    "epochs" : 5,
    "forceCPU" : False
    }

if config["forceCPU"]:
    device = torch.device("cpu")

config["device"] = device

def get_train_and_val_split(df, splitRatio=0.8):
    train=df.sample(frac=splitRatio,random_state=200)
    val=df.drop(train.index)
    print("Number of Training Samples: ", len(train))
    print("Number of Validation Samples: ", len(val))
    return(train, val)

def get_max_length(reviews):
    return len(max(reviews, key=len))

def get_accuracy(logits, labels):
    # get the index of the max value in the row.
    predictedClass = logits.max(dim = 1)[1]

    # get accuracy by averaging over entire batch.
    acc = (predictedClass == labels).float().mean()
    return acc

def trainFunc(net, loss_func, opti, train_loader, test_loader, config):
    best_acc = 0
    for ep in range(config["epochs"]):
        for it, (seq, attn_masks, labels) in enumerate(train_loader):
            opti.zero_grad()
            #seq, attn_masks, labels = seq.cuda(args.gpu), attn_masks.cuda(args.gpu), labels.cuda(args.gpu)
            seq, attn_masks, labels = seq.to(device), attn_masks.to(device), labels.to(device)

            logits = net(seq, attn_masks)
            loss = loss_func(m(logits), labels)

            loss.backward()
            opti.step()
            print("Iteration: ", it+1)

            if (it + 1) % config["printEvery"] == 0:
                acc = get_accuracy(m(logits), labels)
                if not os.path.exists(config["outputFolder"]):
                    os.makedirs(config["outputFolder"])

                # Since a single epoch could take well over hours, we regularly save the model even during evaluation of training accuracy.
                torch.save(net.state_dict(), os.path.join(projectFolder, config["outputFolder"], config["outputFileName"]))
                print("Iteration {} of epoch {} complete. Loss : {} Accuracy : {}".format(it+1, ep+1, loss.item(), acc))
                print("Saving at", os.path.join(projectFolder, config["outputFolder"], config["outputFileName"]))

        # perform validation at the end of an epoch.
        val_acc, val_loss = evaluate(net, loss_func, val_loader, config)
        print(" Validation Accuracy : {}, Validation Loss : {}".format(val_acc, val_loss))
        if val_acc > best_acc:
            print("Best validation accuracy improved from {} to {}, saving model...".format(best_acc, val_acc))
            best_acc = val_acc
            torch.save(net.state_dict(), os.path.join(projectFolder, config["outputFolder"], config["outputFileName"] + "_valTested_" + str(best_acc)))

def evaluate(net, loss_func, dataloader, config):
    net.eval()

    mean_acc, mean_loss = 0, 0
    count = 0

    with torch.no_grad():
        for seq, attn_masks, labels in dataloader:
            #seq, attn_masks, labels = seq.cuda(args.gpu), attn_masks.cuda(args.gpu), labels.cuda(args.gpu)
            seq, attn_masks, labels = seq.to(device), attn_masks.to(device), labels.to(device)

            logits = net(seq, attn_masks)
            mean_loss += loss_func(m(logits), labels)
            mean_acc += get_accuracy(m(logits), labels)
            print("Validation iteration", count+1)
            count += 1

            '''
            The entire validation set was around 0.1 million entries,
            the validationFraction param controls what fraction of the shuffled
            validation set you want to validate the results on.
            '''
            if count > config["validationFraction"] * len(val_set):
                break
    return mean_acc / count, mean_loss / count

class SentimentClassifier(nn.Module):
    def __init__(self, num_classes, device, freeze_bert = True):
        super(SentimentClassifier, self).__init__()
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)
        self.device = device

        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        self.cls_layer = nn.Linear(768, num_classes)

    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        #Feeding the input to BERT model to obtain contextualized representations
        cont_reps, _ = self.bert_layer(seq, attention_mask = attn_masks)

        # Debugging
        print("cont_reps:",cont_reps)
        print('type:',type(cont_reps))
        print("last_hidden_state"==cont_reps)
        # return None

        #Obtaining the representation of [CLS] head
        # cont_reps = cont_reps[0]
        # cls_rep = cont_reps[0]
        cls_rep = cont_reps[:, 0]

        #Feeding cls_rep to the classifier layer
        logits = self.cls_layer(cls_rep)

        return logits.to(self.device)

class AmazonReviewsDataset(Dataset):
    def __init__(self, df, maxlen):
        self.df = df
        # A reset reindexes from 1 to len(df), the shuffled df frames are sparse.
        self.df.reset_index(drop=True, inplace=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',output_hidden_states=True)
        self.maxlen = maxlen

    def __len__(self):
        return(len(self.df))

    def __getitem__(self, index):
        review = self.df.loc[index, 'Text']

        # Classes start from 0.
        label = int(self.df.loc[index, 'Score']) - 1

        # Use BERT tokenizer since it needs to be able to match the tokens to the pre trained words.
        tokens = self.tokenizer.tokenize(review)

        # BERT inputs typically start with a '[CLS]' tag and end with a '[SEP]' tag. For
        tokens = ['[CLS]'] + tokens + ['[SEP]']

        if len(tokens) < self.maxlen:
            # Add the ['PAD'] token
            tokens = tokens + ['[PAD]' for item in range(self.maxlen-len(tokens))]
        else:
            # Truncate the tokens at maxLen - 1 and add a '[SEP]' tag.
            tokens = tokens[:self.maxlen-1] + ['[SEP]']

        # BERT tokenizer converts the string tokens to their respective IDs.
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Converting to pytorch tensors.
        tokens_ids_tensor = torch.tensor(token_ids)

        # Masks place a 1 if token != PAD else a 0.
        attn_mask = (tokens_ids_tensor != 0).long()
        
        return tokens_ids_tensor, attn_mask, label


if __name__ == "__main__":
    print("Configuration is: ", config)

    # Read and shuffle input data.
    
    df = amazon_reviews.load_data()
    print(df.head())

    # file_name = 'Appliances.json.gz'
    # df = pd.read_json(file_name,compression='infer',lines=True).sample(frac=1)
    # df.head(2)

    # df = read_and_shuffle(os.path.join(projectFolder,file_name))
    target_col = 'overall'
    feature_col = 'reviewText'

    df = df[[target_col,feature_col]]
    df.columns = ['Score','Text']
    df.head(2)

    num_classes = df['Score'].nunique()
    print("Number of Target Output Classes:", num_classes)
    totalDatasetSize = len(df)

    print('Loading BERT tokenizer...')
    # config = BertConfig.from_pretrained( 'bert-base-uncased', output_hidden_states=True)    
    # self.bert_model = BertModel.from_pretrained('bert-base-uncased', config=config)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True,output_hidden_states=True)

    # Group by the column Score. This helps you get distribution of the Review Scores.
    symbols = df.groupby('Score')

    scores_dist = []
    for i in range(num_classes):
        scores_dist.append(len(symbols.groups[i+1])/totalDatasetSize)

    train, val = get_train_and_val_split(df, config["splitRatio"])

    val.to_csv("Validations.csv")
    train.to_csv("Train.csv")

    # You can set the length to the true max length from the dataset, I have reduced it for the sake of memory and quicker training.
    #T = get_max_length(reviews)
    T = config["maxLength"]

    train_set = AmazonReviewsDataset(train, T)
    val_set = AmazonReviewsDataset(val, T)

    train_loader = DataLoader(train_set, batch_size = config["batchSize"], num_workers = config["threads"])
    val_loader = DataLoader(val_set, batch_size = config["batchSize"], num_workers = config["threads"])

    # We are unfreezing the BERT layers so as to be able to fine tune and save a new BERT model that is specific to the Sizeable food reviews dataset.
    net = SentimentClassifier(num_classes, config["device"], freeze_bert=False)
    net.to(config["device"])
    weights = torch.tensor(scores_dist).to(config["device"])

    # Setting the Loss function and Optimizer.
    loss_func = nn.NLLLoss(weight=weights)
    opti = optim.Adam(net.parameters(), lr = 2e-5)
    m = nn.LogSoftmax(dim=1)

    torch.cuda.set_device(0)
    trainFunc(net, loss_func, opti, train_loader, val_loader, config)
