import time
import torch
from torch import nn

# https://discuss.pytorch.org/t/preferred-most-efficient-way-of-implementing-pointwise-convolutions/21951/3
# 对于conv1d,要变换number的dimension要放在倒数第二个
# 对于linear,要变换number的dimension要放在最后一个

N = 128
seq_len = 5
embedding_size = 32
use_gpu = torch.cuda.is_available()

input = torch.rand(N, embedding_size, seq_len)  # [128, 32, 5]
if use_gpu:
    input = input.cuda()

c1 = nn.Conv1d(embedding_size, embedding_size, kernel_size=1, padding=0, bias=False)
if use_gpu:
    c1 = c1.cuda()
start_time = time.time()
for it in range(5000):
    output = c1(input)
print('c1 time', time.time() - start_time)
print('output[0,:,0]', output[0, :, 0])

# use linear...

c2 = nn.Linear(embedding_size, embedding_size, bias=False)
if use_gpu:
    c2 = c2.cuda()
c2.weight.data[:] = c1.weight.data.view(embedding_size, embedding_size)
start_time = time.time()
input = input.transpose(-2, -1)
for it in range(5000):
    output = c2(input)
output = output.transpose(-2, -1)
print('c2 time', time.time() - start_time)
print('output[0,:,0]', output[0, :, 0])
