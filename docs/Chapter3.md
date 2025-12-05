# Chapter 3: Importing PyTorch models to torch-mlir
We will now look at how to import PyTorch models to torch-mlir and how it outputs MLIR code in torch dialect and linalg-on-tensor dialect.
Thereby, linalg-on-tensor is combination of linalg, arith and other base dialects.  

A simple model might look like this:
```python
class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return self.linear(x + self.param).clamp(min=0.0, max=1.0)
```

torch-mlir has an export function fx.export_and_import (`torch_mlir.compile()` is deprecated). For that we need to import fx with `from torch_mlir import fx`. This function has a bunch of input parameters but we'll only use 4 of them. Our function call looks like this: 
```python
m = fx.export_and_import(MyModule(), torch.randn(3, 4), output_type=OutputType.TORCH,
                             func_name = "sample_model")
```
We pass an instance of the model, example input parameters, the output type, i.e. the dialect (e.g. TORCH, LINALG_ON_TENSORS), and the function name that we will later use to call the model. We can save the code in a .mlir file. It yiedls:

```
module {
  func.func @main(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,5],f32> {
    %int0 = torch.constant.int 0
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_torch.float32> : tensor<5xf32>) : !torch.vtensor<[5],f32>
    %1 = torch.vtensor.literal(dense_resource<torch_tensor_5_4_torch.float32> : tensor<5x4xf32>) : !torch.vtensor<[5,4],f32>
    %2 = torch.vtensor.literal(dense_resource<torch_tensor_3_4_torch.float32> : tensor<3x4xf32>) : !torch.vtensor<[3,4],f32>
    %int1 = torch.constant.int 1
    %3 = torch.aten.add.Tensor %arg0, %2, %int1 : !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],f32>, !torch.int -> !torch.vtensor<[3,4],f32>
    %4 = torch.aten.transpose.int %1, %int0, %int1 : !torch.vtensor<[5,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,5],f32>
    %5 = torch.aten.mm %3, %4 : !torch.vtensor<[3,4],f32>, !torch.vtensor<[4,5],f32> -> !torch.vtensor<[3,5],f32>
    %6 = torch.aten.add.Tensor %5, %0, %int1 : !torch.vtensor<[3,5],f32>, !torch.vtensor<[5],f32>, !torch.int -> !torch.vtensor<[3,5],f32>
    %7 = torch.aten.clamp %6, %float0.000000e00, %float1.000000e00 : !torch.vtensor<[3,5],f32>, !torch.float, !torch.float -> !torch.vtensor<[3,5],f32>
    return %7 : !torch.vtensor<[3,5],f32>
  }
}
```

And for the LINALG_ON_TENSOR version we get:
``` 
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @sample_model(%arg0: tensor<3x4xf32>) -> tensor<3x5xf32> {
    %cst = arith.constant dense_resource<torch_tensor_3_4_torch.float32> : tensor<3x4xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant dense_resource<torch_tensor_5_4_torch.float32> : tensor<5x4xf32>
    %cst_3 = arith.constant dense_resource<torch_tensor_5_torch.float32> : tensor<5xf32>
    %0 = tensor.empty() : tensor<3x4xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst : tensor<3x4xf32>, tensor<3x4xf32>) outs(%0 : tensor<3x4xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %8 = arith.addf %in, %in_4 : f32
      linalg.yield %8 : f32
    } -> tensor<3x4xf32>
    %2 = tensor.empty() : tensor<4x5xf32>
    %transposed = linalg.transpose ins(%cst_2 : tensor<5x4xf32>) outs(%2 : tensor<4x5xf32>) permutation = [1, 0] 
    %3 = tensor.empty() : tensor<3x5xf32>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%3 : tensor<3x5xf32>) -> tensor<3x5xf32>
    %5 = linalg.matmul ins(%1, %transposed : tensor<3x4xf32>, tensor<4x5xf32>) outs(%4 : tensor<3x5xf32>) -> tensor<3x5xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%5, %cst_3 : tensor<3x5xf32>, tensor<5xf32>) outs(%3 : tensor<3x5xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %8 = arith.addf %in, %in_4 : f32
      linalg.yield %8 : f32
    } -> tensor<3x5xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<3x5xf32>) outs(%3 : tensor<3x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.cmpf ult, %in, %cst_0 : f32
      %9 = arith.select %8, %cst_0, %in : f32
      %10 = arith.cmpf ugt, %9, %cst_1 : f32
      %11 = arith.select %10, %cst_1, %9 : f32
      linalg.yield %11 : f32
    } -> tensor<3x5xf32>
    return %7 : tensor<3x5xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_4_torch.float32: "0x04000000001F723E8C0E683E8CEF4C3F883F2D3EA0A8873E41A1453FCC43033E59C63E3F97F14D3FD5BD223F1AF1163F6D7B313F",
      torch_tensor_5_4_torch.float32: "0x04000000D8A9C13E809B263DFA4EB8BE4635EC3E64B9BB3E80CE3DBC20A895BEB04E46BE5409E2BE7AA7A93E70B931BD6661B03E70C7403E4025CBBD6AB2A3BE7031323E9842A4BE4892E3BDEC2F983EDE668BBE",
      torch_tensor_5_torch.float32: "0x040000006061C8BC30FC8ABD20AE6A3DA27BF63E38A6953E"
    }
  }
#-}
```

Looking at this code we can see that it has no torch ops anymore, so it does only depend on upstream MLIR dialects. With this code, we can continue with the lowering process in the next chapter.