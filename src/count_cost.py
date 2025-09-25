import os
import re
import argparse
import tiktoken

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, required=True)
parser.add_argument("--file", type=str, required=True)
args = parser.parse_args()

id_pattern = r"^\d+\. "
encoding = tiktoken.get_encoding("o200k_base")

prompt = []
prompt_flag = False
input_len = []
output_len = []
time = ""
file_path = os.path.join(args.dir, args.file)
with open(file_path) as file:
    for line in file:
        line = line.strip()
        # prompt
        if line.startswith("Loaded ") and line.endswith("dataset") and "TEST" in line:
            prompt_flag = False
        if prompt_flag:
            prompt.append(line)
        if line == "PROMPT":
            prompt_flag = True
        # input
        if re.search(id_pattern, line) and ("Question: " in line or "Problem: " in line):
            line = re.sub(id_pattern, "", line)
            input_len.append(len(encoding.encode(line)))
        # output
        if line.startswith("Response: "):
            line = line.replace("Response: ", "")
            output_len.append(len(encoding.encode(line)))
        # time
        if line.startswith("Time: "):
            time = line
prompt_len = len(encoding.encode("\n".join(prompt)))
full_input_len = sum(input_len) + len(input_len) * prompt_len
full_output_len = sum(output_len)
mean_input_len = full_input_len / len(input_len)
mean_output_len = full_output_len / len(output_len)
print("Full Input Length: {}".format(full_input_len))
print("Full Output Length: {}".format(full_output_len))
print("Prompt Only Length: {}".format(prompt_len))
print("Mean Input Length: {}".format(mean_input_len))
print("Mean Output Length: {}".format(mean_output_len))
print(time)

print("Done")
