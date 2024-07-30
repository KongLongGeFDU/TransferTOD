import json
import ast
import argparse


def f1_calculator(actual, predict):
    TP, FP, FN = 0, 0, 0
    if len(actual) != 0:
        for g in actual:
            if g in predict:
                TP += 1
            else:
                FN += 1
        for p in predict:
            if p not in actual:
                FP += 1
        precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
        recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
    else:
        if len(predict) == 0:
            precision, recall, F1 = 1, 1, 1
        else:
            precision, recall, F1 = 0, 0, 0
    return F1, recall, precision


def evaluate(actual_file, predict_file):
    actual_data = []
    with open(actual_file, encoding="utf-8") as f1:
        actual_file_data = json.load(f1)
    if isinstance(actual_file_data, dict):
        for domain, conversations in actual_file_data["clean_gpt_human"].items():
            actual_data += conversations
        for domain, conversations in actual_file_data["noise_gpt_human"].items():
            actual_data += conversations
    else:
        actual_data = actual_file_data
    predict_data = []
    with open(predict_file, encoding="utf-8") as f2:
        for line in f2:
            predict_data.append(json.loads(line))
    total = 0

    joint_acc, F1_pred = 0, 0
    for i in range(len(predict_data)):
        print("-" * 20)
        print(i + 1)
        try:
            print(actual_data[i]["conversations"][0]["value"] + "\n")
            print("actual_data: " + actual_data[i]["conversations"][1]["value"].replace("'", "\"").replace("None", "null").replace("True", "true").replace("False", "false"))
            print("predict_data: " + predict_data[i]["conversations"][1]["value"].replace("'", "\"").replace("None", "null").replace("True", "true").replace("False", "false"))
        except UnicodeEncodeError as e:
            print(e)
            continue
        try:
            actual_slots = json.loads(actual_data[i]["conversations"][1]["value"][actual_data[i]["conversations"][1]["value"].find("{"):actual_data[i]["conversations"][1]["value"].find("}") + 1].replace("'", "\"").replace("None", "null").replace("True", "true").replace("False", "false"))
            predict_slots = json.loads(predict_data[i]["conversations"][1]["value"][predict_data[i]["conversations"][1]["value"].find("{"):predict_data[i]["conversations"][1]["value"].find("}") + 1].replace("'", "\"").replace("None", "null").replace("True", "true").replace("False", "false"))
        except SyntaxError as e:
            print(e)
            continue
        except ValueError as e:
            print(e)
            continue
        except json.decoder.JSONDecodeError as e:
            print(e)
            continue
        try:
            actual = set([(k, v) for k, v in actual_slots.items()])
            predict = set([(k, v) for k, v in predict_slots.items()])
        except TypeError as e:
            print(e)
            continue
        total += 1
        if actual == predict:
            joint_acc += 1
            print("--JointAcc Count--")
        temp_f1, temp_r, temp_p = f1_calculator(actual, predict)
        F1_pred += temp_f1
        print("--F1_pred: {}--".format(temp_f1))

    print("\n" + "-" * 20)
    print("Total: {}".format(total))
    total = len(predict_data)

    joint_acc_score = joint_acc / float(total) if total != 0 else 0
    F1_score = F1_pred / float(total) if total != 0 else 0
    print("Joint_acc_score: {:.3%}".format(joint_acc_score))
    print("F1_score: {:.3%}".format(F1_score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--actual_file", type=str)
    parser.add_argument("-p", "--predict_file", type=str)
    args = parser.parse_args()
    evaluate(args.actual_file, args.predict_file)
