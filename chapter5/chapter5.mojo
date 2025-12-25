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
from math import align_up
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
    bx = Int(block_dim.x)
    by = Int(block_dim.y)

    c_col = bx * BN + tx
    c_row = by * BN + ty

    c_val: Scalar[dtype] = 0

    for k in range(0, K, BK):
        smem_a[ty, tx] = a.load[1](c_row, k + tx)
        smem_b[ty, tx] = b.load[1](k + ty, c_col)

        barrier()

        for bk in range(BK):
            c_val += smem_a.load[1](ty, bk) * smem_b.load[1](bk, tx)

    c.store[1](c_row, c_col, c_val)


def equal[
    dtype: DType, a_layout: Layout, b_layout: Layout
](
    a: LayoutTensor[dtype, a_layout, MutAnyOrigin],
    b: LayoutTensor[dtype, b_layout, MutAnyOrigin],
) -> Bool:
    is_equal = False

    @__copy_capture(a, b)
    @always_inline
    @parameter
    fn equals[width: Int, rank: Int, alignment: Int = 1](index: IndexList[rank]):
        is_equal &= a.load[width](index) == b.load[width](index)

    elementwise[equals, 1](a.runtime_layout.shape.value)
    return is_equal
    

def main():
    comptime M = 256
    comptime N = 256
    comptime K = 256
    comptime BM = 32
    comptime BN = 32
    comptime BK = 32

    comptime dtype = DType.float32
    var ctx = DeviceContext()

    var a_h = ctx.enqueue_create_host_buffer[dtype](M * K)
    var b_h = ctx.enqueue_create_host_buffer[dtype](K * N)
    var c_h = ctx.enqueue_create_host_buffer[dtype](M * N)

    var c_ref = ctx.enqueue_create_host_buffer[dtype](M * N)

    ctx.synchronize()

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
        grid_dim=(N // BN, M // BM),
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

    if equal(c_ref_tensor, c_host_tensor):
        print("Pass")
    else:
        print("Fail")
