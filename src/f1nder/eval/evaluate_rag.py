#Usiamo sempre il modello Gemma per valutare la correttezza delle
#risposte fornite nella fase precedente. In questo modo è possibile
#valutare positivamente anche risposte corrette ma non identiche.

from argparse import ArgumentParser
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

judging_prompt ="""
You are a strict and impartial evaluator.

Determine whether the Generated Answer is fully equivalent to the Ground Truth Answer.

User Question:
"{question}"

Ground Truth Answer (authoritative):
"{ground_truth}"

Generated Answer:
"{generated_answer}"

Steps:
1. Extract the key factual statements from the Ground Truth.
2. Check whether each statement is present and correctly expressed in the Generated Answer.
3. Identify any additional claims in the Generated Answer not supported by the Ground Truth.
4. Decide equivalence.

Decision Rules:
- Equivalent = YES only if:
  a) No key information from the Ground Truth is missing, AND
  b) No extra unsupported or incorrect information is added.
- Otherwise, Equivalent = NO.

Output Format (You MUST respect this output format):
TRUE if the answers are equivalent
FALSE otherwise
"""

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

    for elem in ground_truth_answers[:100]:
        ground_truth = elem['answer']
        id = elem['query_id']

        #search model_answer by query_id
        model_answer = list(filter(lambda e: e['query_id'] == id, model_answers))

        #check if a model answer is present for query_id
        if(model_answer):
            answer = model_answer[0]['answer']
            org_question = model_answer[0]['question']
            print(ground_truth, " ", answer)

            prompt = judging_prompt.format(
                question = org_question,
                ground_truth = ground_truth,
                generated_answer = answer
            )
            print(prompt)
            messages = [
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