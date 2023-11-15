import torch
import numpy as np
## khởi  tạo
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

print(x_data)


np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(np_array)

x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

shape = (2,3)
rand_tensor = torch.rand((5,3))
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}. device {zeros_tensor.device} \n")


tensor = torch.ones(2,3)
print(tensor.shape)
t1 = torch.cat([3*tensor, 2*tensor, tensor], dim=-1) ### 0 là nối theo hàng,  1 nối theo cột , -1 nối theo chiều sâu
print(t1.shape)
print(t1)

agg = tensor.sum()
agg_item = agg.item() ## chuyển từ dạng tensor sang python để dùng được cho thao tác khác
print(agg_item, type(agg_item))


## tự lưu luôn và thay đổi luôn tensor
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)


### Nhân ma trận 
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T  ### Nhân ma trận với ma trận chuyển vị
y2 = tensor.matmul(tensor.T) ### Nhân ma trận 

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)  ### Nhân ma trận với ma trận chuyển vị rồi lưu vào y3


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)   ### Nhân element ( tích chập ) ma trận với ma trận 

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)  ### Nhân element ( tích chập ) ma trận với ma trận rồi lưu vào z3