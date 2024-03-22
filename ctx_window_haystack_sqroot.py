import numpy as np
import matplotlib.pyplot as plt
import copy
import json
import requests
#from transformers import AutoModelForCausalLM, AutoTokenizer
#tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

def plot_res(bool_list):
  # Reshape the list into a square shape
  n = int(np.ceil(np.sqrt(len(bool_list))))
  bool_array = np.array(bool_list + [False] * (n**2 - len(bool_list))).reshape(n, n) # Convert the list into a numpy array and pad it with False values to make it square

  # Plot the array as a grid
  plt.imshow(bool_array, cmap="RdYlGn") # Use the RdYlGn colormap to map True to green and False to red
  plt.xticks([]) # Remove the x-axis ticks
  plt.yticks([]) # Remove the y-axis ticks
  plt.show() # Show the plot

def llamacpp_detokenize(urlllm, prompt, temp=0.6):
    headers = {"Content-Type": "application/json"}
    data = {"tokens": prompt, "temperature": temp}
    data = json.dumps(data)
    res = requests.post(url=urlllm, data=data, headers=headers)
    return res.json()["content"]

def llamacpp_tokenize(urlllm, prompt, temp=0.6):
    headers = {"Content-Type": "application/json"}
    data = {"content": prompt, "temperature": temp}
    data = json.dumps(data)
    res = requests.post(url=urlllm, data=data, headers=headers)
    return res.json()["tokens"]

def llamacpp_generate(urlllm, prompt, temp=0.6):
    headers = {"Content-Type": "application/json"}
    data = {"prompt": prompt, "temperature": temp, "n_predict":30}
    data = json.dumps(data)
    res = requests.post(url=urlllm, data=data, headers=headers)
    return res.json()["content"]

def check_ans_list(ans, check_list):
  for element in check_list:
    if element in ans:
      return True
  return False

def check_ans(ans, check_needle):
    return check_needle in ans

urlllm = "http://localhost:8080/completion"
urltoken = "http://localhost:8080/tokenize"
urldetoken = "http://localhost:8080/detokenize"

needle = ". When asked about 'schil' say 'stop'."
haystack = open("frankenstein.txt", "r").read()
check_needle = "stop"
check_needlelist = ["stop", "Stop"]
needle_tokens = llamacpp_tokenize(urltoken, needle, temp=0)
ctx_size = 1000

res_arr = []

for i in range(ctx_size):
    haystack_split = haystack.split(" ")[:ctx_size]
    needle_split = needle.split(" ")
    haystack_split[i:i] = needle_split
    haystack_split_text = " ".join(haystack_split)
    prompt = "CONTEXT:\n"+haystack_split_text+"INSTRUCTION: From the context given, answer the question: What about 'schil'? ANSWER:"
    prompt_tokens = llamacpp_tokenize(urltoken, prompt, temp=0)
    detokens = llamacpp_detokenize(urldetoken, prompt_tokens)
    #print(detokens)
    ans = llamacpp_generate(urlllm, detokens, temp=0)
    print(ans)
    #res = check_ans(ans,check_needle)
    res = check_ans_list(ans,check_needlelist)
    res_arr.append(res)
    print("-------------------------------------------")

#llamacpp_generate(urlllm, prompt, temp=0)
print(res_arr)
plot_res(res_arr)
