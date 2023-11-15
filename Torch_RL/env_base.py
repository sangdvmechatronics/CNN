from collections import defaultdict  ### Chức năng chính là giúp quản lý thông tin về các trạng thái trong quá trình tương tác với môi trường.
from typing import Optional ###  Thường được sử dụng trong các hàm và phương thức để biểu diễn việc có hay không có giá trị trả về.

import numpy as np
import torch
import tqdm ### Thư viện để tạo thanh tiến trình, giúp theo dõi tiến trình của một vòng lặp.
from tensordict.nn import TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase
from torch import nn  

from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec   ### Định nghĩa các loại tensor specifications (Spec) 
### được sử dụng để mô tả các không gian trạng thái và hành động trong môi trường
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp

DEFAULT_X = np.pi
DEFAULT_Y = 1.0


#### Hàm kiểm soát hành động để chuyển trạng thái, thực hiện một bước để tương tác với môi trường
## EnvBase.step() nhận vào  một hành động, sau đó môi trường sẽ thực hiện bước tương tác 
# bằng cách thực hiện hành động này và trả về thông tin về trạng thái kế tiếp và phần thưởng nhận được.
## Định nghĩa như sau: 
def _step(tensordict):
    th, thdot = tensordict["th"], tensordict["thdot"]  # th := theta

    g_force = tensordict["params", "g"]
    mass = tensordict["params", "m"]
    length = tensordict["params", "l"]
    dt = tensordict["params", "dt"]
    u = tensordict["action"].squeeze(-1)
    u = u.clamp(-tensordict["params", "max_torque"], tensordict["params", "max_torque"]) ### Giới hạn giá trị hành động để đảm bảo rằng
    ### nó không vượt quá ngưỡng tối đa.
    costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2) ### Tính toán  phần thưởng chi phí dựa trên giá trị của "th", "thdot", và "u".

    new_thdot = (
        thdot
        + (3 * g_force / (2 * length) * th.sin() + 3.0 / (mass * length**2) * u) * dt
    )
    new_thdot = new_thdot.clamp(
        -tensordict["params", "max_speed"], tensordict["params", "max_speed"]
    )
    new_th = th + new_thdot * dt
    reward = -costs.view(*tensordict.shape, 1) ### ính toán phần thưởng và trả về dưới dạng tensor với kích thước phù hợp.
    done = torch.zeros_like(reward, dtype=torch.bool)
    out = TensorDict(        ####   out mới chứa các thông tin cập nhật sau mỗi bước thời gian.
        {
            "th": new_th,
            "thdot": new_thdot,
            "params": tensordict["params"],
            "reward": reward,
            "done": done,
        },
        tensordict.shape,
    )
    return out


def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi   ### Chuẩn hóa góc x về khoảng [-pi, pi].

#### Đặt lại vòng lặp thửu sai EnvBase._reset(), kiểm tra để bắt đầu đào tạo eposide


def _reset(self, tensordict):
    if tensordict is None or tensordict.is_empty(): ## kiểm tra xem có tham số trong dict không, nếu không có thì sẽ khởi tạo
        tensordict = self.gen_params(batch_size=self.batch_size) ## tạo ra các siêu tham số cho môi trường

    high_th = torch.tensor(DEFAULT_X, device=self.device) ### Sử dụng để giới hạn các giá trị trạng thái
    high_thdot = torch.tensor(DEFAULT_Y, device=self.device)
    low_th = -high_th
    low_thdot = -high_thdot

    th = ( #### Tạo giá trị ngẫu nhiên cho các thành phần th và thdot của trạng thái
           ### dựa trên các giới hạn đã xác định trước đó.
        torch.rand(tensordict.shape, generator=self.rng, device=self.device)
        * (high_th - low_th)
        + low_th
    )
    thdot = (
        torch.rand(tensordict.shape, generator=self.rng, device=self.device)
        * (high_thdot - low_thdot)
        + low_thdot
    )
    out = TensorDict(
        {
            "th": th,
            "thdot": thdot,
            "params": tensordict["params"],
        },
        batch_size=tensordict.shape,
    )
    return out

## Các tham số siêu dữ liệu môi trường metadata nhằm mô tả môi trường học tăng cường
## đặc tả liên quan tới đầu ra và đầu vào môi trường
## có 4 thông số cần mã hóa :
# EnvBase.observation_spec: mô tả không gian quan sát trong môi trường
#  EnvBase.action_spec: mô tả không gian hành động tương ứung với action trong dict
# EnvBase.reward_spec: không gian phần thường
# EnvBase.done_spec: không gian của trạng thái kết thúc



##  khởi tạo các đặc điểm (specs) cho môi trường học tăng cường.
# Điều này bao gồm các đặc điểm cho quan sát (observations), hành động (actions), và các thông số khác như thưởng (rewards).

def _make_spec(self, td_params):
    # Define quan sát từ môi trường
    self.observation_spec = CompositeSpec(
        th=BoundedTensorSpec(
            low=-torch.pi,
            high=torch.pi,
            shape=(),
            dtype=torch.float32,
        ),
        thdot=BoundedTensorSpec(
            low=-td_params["params", "max_speed"],
            high=td_params["params", "max_speed"],
            shape=(),
            dtype=torch.float32,
        ),
 
        params=make_composite_from_td(td_params["params"]),
        shape=(),
    )
 
    self.state_spec = self.observation_spec.clone() ## thiết lập với môi trường không giữ trạng thái stateless env, môi trường sẽ không lưu
    # trữ trạng thái trước đó mà chỉ phụ thuộc vào trạng thái và hđ hiện tại

    self.action_spec = BoundedTensorSpec(
        low=-td_params["params", "max_torque"],
        high=td_params["params", "max_torque"],
        shape=(1,),
        dtype=torch.float32,
    )
    self.reward_spec = UnboundedContinuousTensorSpec(shape=(*td_params.shape, 1)) ## định nghĩa phần thường ( đặt là không có giới hạn)


def make_composite_from_td(td):  ### định rõ không gian của quan sát, hành động, trạng thái và thưởng trong môi trường học tăng cường.
  
    composite = CompositeSpec(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else UnboundedContinuousTensorSpec(
                dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
            )
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite

def _set_seed(self, seed: Optional[int]):
    rng = torch.manual_seed(seed)
    self.rng = rng



## Define các tham số cho môi trường ( các tham số ràng buộc )
def gen_params(g=10.0, batch_size=None) -> TensorDictBase:
    """Returns a ``tensordict`` containing the physical parameters such as gravitational force and torque or speed limits."""
    if batch_size is None:
        batch_size = []
    td = TensorDict(
        {
            "params": TensorDict(
                {
                    "max_speed": 8,
                    "max_torque": 2.0,
                    "dt": 0.05,
                    "g": g,
                    "m": 1.0,
                    "l": 1.0,
                },
                [],
            )
        },
        [],
    )
    if batch_size:
        td = td.expand(batch_size).contiguous()
    return td

## Ghép hoàn chỉnh môi trường

class PendulumEnv(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    batch_locked = False

    def __init__(self, td_params=None, seed=None, device="cpu"):
        if td_params is None:
            td_params = self.gen_params()

        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    # Helpers: _make_step and gen_params
    gen_params = staticmethod(gen_params)
    _make_spec = _make_spec

    # Mandatory methods: _step, _reset and _set_seed
    _reset = _reset
    _step = staticmethod(_step)
    _set_seed = _set_seed


## Kiểm tra xem môi trường đã hoàn chỉnh chưa 
env = PendulumEnv()
check_env_specs(env)
## check các thành phân môi trường 
# print("observation_spec:", env.observation_spec)
# print("state_spec:", env.state_spec)
# print("reward_spec:", env.reward_spec)


## thực hiện các lệnh kiểm tra khác 
td = env.reset()
print("reset tensordict", td)
td = env.rand_step(td)
print("random step tensordict", td)