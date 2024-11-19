import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def main():
    """"""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    model_options = ["microsoft/Phi-3-medium-128k-instruct", "microsoft/Phi-3-mini-128k-instruct", "facebook/opt-2.7b",
                     "facebook/opt-6.7b",  "infly/OpenCoder-8B-Instruct"]
    
    for m in model_options:
        print(m)

    model_name = input("Choose a model (must be from transformers like facebook/opt-350m): ")
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    while True:    
        inpt = input("Give an input for the model: ")
        if inpt.lower() == 'x' or inpt.lower() == 'q':
            break
        token_inpt = int(input("How many tokens would you like to generate? "))
        input_ids = tokenizer(inpt, return_tensors="pt").input_ids.to(device)

        outputs = model.generate(input_ids, max_new_tokens=token_inpt)
        output = tokenizer.decode(outputs[0])
        print(output)
    
if __name__ == "__main__":
    main()
