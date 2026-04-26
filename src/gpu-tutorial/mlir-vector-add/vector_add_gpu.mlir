#ceildiv128 = affine_map<(d0) -> (d0 ceildiv 128)>

module attributes {gpu.container_module} {
  func.func @vector_add(%a: memref<?xf32>, %b: memref<?xf32>,
                        %out: memref<?xf32>, %n: index) {
    %c1 = arith.constant 1 : index
    %block = arith.constant 128 : index
    %grid = affine.apply #ceildiv128(%n)

    %d_a = gpu.alloc (%n) : memref<?xf32, 1>
    %d_b = gpu.alloc (%n) : memref<?xf32, 1>
    %d_out = gpu.alloc (%n) : memref<?xf32, 1>

    gpu.memcpy %d_a, %a : memref<?xf32, 1>, memref<?xf32>
    gpu.memcpy %d_b, %b : memref<?xf32, 1>, memref<?xf32>

    gpu.launch blocks(%bx, %by, %bz) in (%gx = %grid, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %block, %sy = %c1, %sz = %c1) {
      %base = arith.muli %bx, %block : index
      %i = arith.addi %base, %tx : index
      %inside = arith.cmpi ult, %i, %n : index
      scf.if %inside {
        %av = memref.load %d_a[%i] : memref<?xf32, 1>
        %bv = memref.load %d_b[%i] : memref<?xf32, 1>
        %sum = arith.addf %av, %bv : f32
        memref.store %sum, %d_out[%i] : memref<?xf32, 1>
      }
      gpu.terminator
    }

    gpu.memcpy %out, %d_out : memref<?xf32>, memref<?xf32, 1>

    gpu.dealloc %d_a : memref<?xf32, 1>
    gpu.dealloc %d_b : memref<?xf32, 1>
    gpu.dealloc %d_out : memref<?xf32, 1>
    return
  }
}
