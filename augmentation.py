import json
import random
import pandas as pd


def main():
    
    with open("data/train.json") as f:
        data = json.load(f)
    print ("Original data:", len(data))
    
    question = []
    text = []
    label = []

    for d in data:
        question.append(d["question"])
        text.append(d["title"] + " . " +d["text"])
        label.append(int(d["label"]))

    df = pd.DataFrame()
    df["question"] = question
    df["text"] = text
    df["label"] = label
    
    
    for i in range(len(data)):
        q = question[i]
        t = text[i].strip()
        l = label[i]

        t = t.split(" . ")
        t = [x.strip(".") for x in t]

        if len(t) == 3:
            t = [t[0], t[2], t[1]]
            question.append(q)
            text.append(" . ".join(t))
            label.append(l)
        elif len(t) > 3:
            current_text = [" . ".join(t) + " ."]

            i = 0
            max_i = 5 if l == 1 else 2
            while (True):

                if i == max_i:
                    break
                temp_1 = t[1:]
                random.shuffle(temp_1)
                temp_2 = [t[0]]
                temp_2.extend(temp_1)
                temp_text = " . ".join(temp_2) + " ."

                if temp_text not in current_text:
                    question.append(q)
                    text.append(temp_text)
                    label.append(l)
                    current_text.append(temp_text)
                    i += 1

    df = pd.DataFrame()
    df["question"] = question
    df["text"] = text
    df["label"] = label
    print ("Augmented data:", len(df))

    question_ratio = []
    for q in list(set(question)):
        temp_df = df[df["question"] == q]
        ratio = sum(temp_df["label"]) / len(temp_df["label"])
        question_ratio.append((q, ratio))
    question_ratio = sorted(question_ratio, key=lambda x: x[1])

    train_question = []
    valid_question = []

    for i, q in enumerate(question_ratio):
        if (i % 5) == 0:
            valid_question.append(q[0])
        else:
            train_question.append(q[0])

            
    train_df = df[df['question'].isin(train_question)]
    train_df.to_csv("data/train.csv", index=False)
    print ("Train data:", len(data))
    

    valid_df = df[df['question'].isin(valid_question)]
    valid_df.to_csv("data/valid.csv", index=False)
    print ("Valid data:", len(data))


if __name__ == "__main__":
    main()
