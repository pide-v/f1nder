#Usiamo sempre il modello Gemma per valutare la correttezza delle
#risposte fornite nella fase precedente. In questo modo è possibile
#valutare positivamente anche risposte corrette ma non identiche.

from argparse import ArgumentParser
import json
import re
import string
import unicodedata
from transformers import AutoTokenizer, AutoModelForCausalLM

system_prompt = """
You are an evaluator.

Your task is to determine whether two answers to the same question convey the same meaning.

Answer YES if the answers are semantically equivalent, even if they use different wording.
Answer NO if they differ in meaning, provide conflicting information, omit key details, or change the intent.

Ignore differences in style, length, or wording.
Base your decision only on factual content.

Respond using only YES or NO. Do not provide explanations.
""".strip()

user_prompt = """
Pair 1:
Question: {question}
Answer: {ground_truth}

Pair 2:
Question: {question}
Answer: {generated_answer}
""".strip()


def normalize_text(text):
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def evaluate(test_queries_answers_path, model_answers_path, output_path):
    #load ground truth and model answers
    with open(test_queries_answers_path, "r") as f:
        ground_truth_answers = json.load(f)

    model_answers = []
    with open(model_answers_path, "r") as f:
        for line in f:
            model_answers.append(json.loads(line))

    #model definition
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")

    for elem in ground_truth_answers:
        ground_truth = elem['answer']
        id = elem['query_id']

        #search model_answer by query_id
        model_answer = list(filter(lambda e: e['query_id'] == id, model_answers))

        #check if a model answer is present for query_id
        if(model_answer):
            answer = model_answer[0]['answer']
            org_question = model_answer[0]['question']

            #It is crucial to normalize text
            prompt = user_prompt.format(
                question=normalize_text(org_question),
                ground_truth=normalize_text(ground_truth),
                generated_answer=normalize_text(answer)
            )

            print(prompt)

            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            outputs = model.generate(**inputs, max_new_tokens=40)
            feedback = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
                )
            print(feedback)

            obj = {
                "feedback": feedback,
                "query_id": id
            }
            with open(output_path, "a") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                f.flush()

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--test_queries_answers_path")
    ap.add_argument("--model_answers_path")
    ap.add_argument("--output_path")
    args = ap.parse_args()

    evaluate(
        args.test_queries_answers_path,
        args.model_answers_path,
        args.output_path
        )