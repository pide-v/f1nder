import json
from argparse import ArgumentParser

def evaluate_qa(
        top1_path,
        top5_path,
        output_path
):
    top1_answers = []
    with open(top1_path, "r") as f:
        for line in f:
            top1_answers.append(json.loads(line)["feedback"].strip())
    top5_answers = []
    with open(top5_path, "r") as f:
        for line in f:
            top5_answers.append(json.loads(line)["feedback"].strip())

    top1_acc = sum(x == "YES" for x in top1_answers) / len(top1_answers)
    top5_acc = sum(x == "YES" for x in top5_answers) / len(top5_answers)


    print(f"top1 acc: {top1_acc}\ntop5 acc: {top5_acc}")
      

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--top1_path")
    ap.add_argument("--top5_path")
    ap.add_argument("--output_path")
    args = ap.parse_args()

    evaluate_qa(
        args.top1_path,
        args.top5_path,
        args.output_path
        )