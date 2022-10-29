# import library
import torch
  
# create two 2D tensors
first = torch.Tensor([[7, -2, 3],
                      [29, 9, -5],
                      [2, -8, 34],
                      [24, 62, 98]])
  
second = torch.Tensor([[7, -5, 3],
                       [26, 9, -4],
                       [3, -8, 43],
                       [23, -62, 98]])
third  = torch.Tensor([[8, -5, 3],
                       [24, 9, -4],
                       [3, -5, 41],
                       [23, -62, 98]])
  
# print first tensors
print("First Tensor:", first)
  
# print second tensors
print("Second Tensor:\n", second)
  
  
print("After Comparing Both Tensors")
  
# Compare element wise tensors first
# and second
t1 = torch.eq(first, second).double()
t2 = torch.eq(third, second).double()
t3 = torch.eq(t1, t2).double()

print(t1)
print(t2)
print(t3)

