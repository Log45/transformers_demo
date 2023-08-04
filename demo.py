import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def main():
    """"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_name = input("Choose a model (must be from transformers like facebook/opt-350m): ")
    
    inpt = input("Give an input for the model: ")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    input_ids = tokenizer(inpt, return_tensors="pt").input_ids.to(device)

    outputs = model.generate(input_ids, max_new_tokens=50)
    output = tokenizer.decode(outputs[0])
    print(output)
    
if __name__ == "__main__":
    main()