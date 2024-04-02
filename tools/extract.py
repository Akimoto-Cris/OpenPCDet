# Given input is a list of tensor strings with values. The goal is to extract just the numerical values from these strings.

input_tensors = [
    tensor(0.7750, device='cuda:0'), tensor(0.9970, device='cuda:0'), tensor(0.9700, device='cuda:0'), tensor(0.8820, device='cuda:0'), tensor(0.9990, device='cuda:0'), tensor(0.9970, device='cuda:0'), tensor(0.9770, device='cuda:0'), tensor(0.9650, device='cuda:0'), tensor(0.7030, device='cuda:0'), tensor(0.6390, device='cuda:0')
]

# Extracting numerical values using list comprehension
extracted_values = [float(tensor.split('(')[1].split(',')[0]) for tensor in input_tensors]

print(extracted_values)
