// RUN: tutorial-opt <%s -linalg-to-bufferization -convert-matmul-to-blas -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: llvm.func @cblas_sgemm(i32, i32, i32, i32, i32, i32, f32, !llvm.ptr, i32, !llvm.ptr, i32, f32, !llvm.ptr, i32)
// CHECK-LABEL: func.func @foo(
// CHECK: llvm.call @cblas_sgemm(
// CHECK: bufferization.clone
// CHECK: return

func.func @foo(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>, %acc: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result = linalg.matmul
    ins(%lhs, %rhs: tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%acc: tensor<?x?xf32>)
  -> tensor<?x?xf32>
  return %result: tensor<?x?xf32>
}
