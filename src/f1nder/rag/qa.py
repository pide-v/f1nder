import json
from argparse import ArgumentParser
from prompts import rag_system_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_test(test_queries_path, test_queries_answers_path):
    with open(test_queries_path, "r") as f:
        test = json.load(f)
    with open(test_queries_answers_path, "r") as f:
        answers = json.load(f)
    return test, answers


def qa(test_queries_path, test_queries_answers_path, output_path):
    test, answers = load_test(test_queries_path, test_queries_answers_path)

    #Caricamento modello e tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")

    #Ciclo su tutte le domande del test
    #Poi dovranno arrivare documenti da bm25f

    #10'000 sono davvero tante da runnare in locale, valutiamo se fare un sottoinsieme
    for t in test[:3]:
        question = t['question']

        #Preparazione del prompt per RAG
        rag_prompt = rag_system_prompt.format(
            document = "***Document***",
            question = question
        )
        print(rag_prompt)
        messages = [
            {
                "role": "user",
                "content": rag_prompt
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
        answer = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
            )

        #Salvataggio su file della risposta per ciascun query_id
        obj = {
            "query_id": t['query_id'],
            "answer": answer
        }
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f.flush()

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--test_queries_path")
    ap.add_argument("--test_queries_answers_path")
    ap.add_argument("--output_path")
    args = ap.parse_args()

    qa(
        args.test_queries_path,
        args.test_queries_answers_path,
        args.output_path
        )