from algorithm.functional import elementwise
from gpu import (
    thread_idx,
    block_idx,
    global_idx,
    block_dim,
    grid_dim,
    barrier,
    lane_id,
    WARP_SIZE,
)
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor, print_layout
from math import align_up, ceildiv
from random.random import randn
from utils.index import IndexList

fn ref_matmul[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    M: Int,
    N: Int,
    K: Int,
](
    a: LayoutTensor[dtype, a_layout, ImmutAnyOrigin],
    b: LayoutTensor[dtype, b_layout, ImmutAnyOrigin],
    c: LayoutTensor[dtype, c_layout, MutAnyOrigin],
):
    for m in range(M):
        for n in range(N):
            c_val: Scalar[dtype] = 0
            for k in range(K):
                c_val += a.load[1](m, k) * b.load[1](k, n)
            c.store[1](m, n, c_val)


fn matmul[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    M: Int,
    N: Int,
    K: Int,
    BM: Int,
    BN: Int,
    BK: Int,
](
    a: LayoutTensor[dtype, a_layout, ImmutAnyOrigin],
    b: LayoutTensor[dtype, b_layout, ImmutAnyOrigin],
    c: LayoutTensor[dtype, c_layout, MutAnyOrigin],
):
    smem_a = LayoutTensor[
        dtype,
        Layout.row_major(BM, BK),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    smem_b = LayoutTensor[
        dtype,
        Layout.row_major(BK, BN),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    tx = Int(thread_idx.x)
    ty = Int(thread_idx.y)
    bx = Int(block_idx.x)
    by = Int(block_idx.y)

    c_col = bx * BN + tx
    c_row = by * BM + ty

    c_val: Scalar[dtype] = 0

    for k in range(0, K, BK):
        smem_a_val: Scalar[dtype] = 0
        smem_b_val: Scalar[dtype] = 0

        if c_row < M and k + tx < K:
            smem_a_val = a.load[1](c_row, k + tx)

        if k + ty < K and c_col < N:
            smem_b_val = b.load[1](k + ty, c_col)

        smem_a[ty, tx] = smem_a_val
        smem_b[ty, tx] = smem_b_val

        barrier()

        for bk in range(BK):
            c_val += smem_a.load[1](ty, bk) * smem_b.load[1](bk, tx)

        barrier()

    if c_row < M and c_col < N:
        c.store[1](c_row, c_col, c_val)


def equal[
    dtype: DType, a_layout: Layout, b_layout: Layout
](
    out_tensor: LayoutTensor[dtype, b_layout, MutAnyOrigin],
    ref_tensor: LayoutTensor[dtype, a_layout, MutAnyOrigin],
) -> Bool:
    shape = ref_tensor.runtime_layout.shape.value

    for m in range(shape[0]):
        for n in range(shape[1]):
            out_val = out_tensor.load[1](m, n)
            ref_val = ref_tensor.load[1](m, n)

            if out_val != ref_val:
                print(m, n)
                print("Out: ", out_val)
                print("Ref: ", ref_val)
                return False
    return True
    

def main():
    comptime M = 127
    comptime N = 256
    comptime K = 256
    comptime BM = 16
    comptime BN = 16
    comptime BK = 16

    comptime dtype = DType.float32
    var ctx = DeviceContext()

    var a_h = ctx.enqueue_create_host_buffer[dtype](M * K)
    var b_h = ctx.enqueue_create_host_buffer[dtype](K * N)
    var c_h = ctx.enqueue_create_host_buffer[dtype](M * N)

    var c_ref = ctx.enqueue_create_host_buffer[dtype](M * N)

    randn[dtype](a_h.unsafe_ptr(), len(a_h))
    randn[dtype](b_h.unsafe_ptr(), len(b_h))

    var a_d = ctx.enqueue_create_buffer[dtype](M * K)
    var b_d = ctx.enqueue_create_buffer[dtype](K * N)
    var c_d = ctx.enqueue_create_buffer[dtype](M * N)

    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(b_d, b_h)

    ctx.synchronize()

    comptime a_layout = Layout.row_major(M, K)
    comptime b_layout = Layout.row_major(K, N)
    comptime c_layout = Layout.row_major(M, N)

    a_tensor = LayoutTensor[dtype, a_layout, ImmutAnyOrigin](a_d)
    b_tensor = LayoutTensor[dtype, b_layout, ImmutAnyOrigin](b_d)
    c_tensor = LayoutTensor[dtype, c_layout, MutAnyOrigin](c_d)

    comptime kernel = matmul[
        dtype, a_layout, b_layout, c_layout, M, N, K, BM, BN, BK
    ]

    ctx.enqueue_function_checked[kernel, kernel](
        a_tensor,
        b_tensor,
        c_tensor,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(BN, BM),
    )

    a_ref_tensor = LayoutTensor[dtype, a_layout, ImmutAnyOrigin](a_h)
    b_ref_tensor = LayoutTensor[dtype, b_layout, ImmutAnyOrigin](b_h)
    c_ref_tensor = LayoutTensor[dtype, c_layout, MutAnyOrigin](c_ref)

    ref_matmul[dtype, a_layout, b_layout, c_layout, M, N, K](
        a_ref_tensor, b_ref_tensor, c_ref_tensor
    )

    ctx.enqueue_copy(c_h, c_d)
    ctx.synchronize()

    c_host_tensor = LayoutTensor[dtype, c_layout, MutAnyOrigin](c_h)

    if equal(c_host_tensor, c_ref_tensor):
        print("Pass")
    else:
        print("Fail")
