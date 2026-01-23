echo 'module {
  func.func @torch_constant_int() -> !torch.bool {
    %int_1 = torch.constant.bool true
    return %int_1 : !torch.bool
  }
}' | torch-mlir-opt \
  -pass-pipeline="builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)" \
  -mlir-print-ir-after-all -mlir-disable-threading
