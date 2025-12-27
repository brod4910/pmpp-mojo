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


fn conv1d_ref[
    dtype: DType,
    layout: Layout,
    filter_layout: Layout,
    M: Int,
    radius: Int,
    filter_size: Int,
](
    a: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    filter: LayoutTensor[dtype, filter_layout, ImmutAnyOrigin],
    c: LayoutTensor[dtype, layout, MutAnyOrigin],
):

    for m in range(M):
        out_val: Scalar[dtype] = 0
        for r in range(-radius, radius):
            mf = m + r
            if mf >= 0 and mf < M:
                f = filter.load[1](0, r + radius)
                in_val = a.load[1](0, mf)
                out_val += in_val * f

        c.store[1](0, m, out_val)


fn conv_1d_basic[
    dtype: DType,
    layout: Layout,
    filter_layout: Layout,
    M: Int,
    radius: Int,
    filter_size: Int,
](
    a: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    filter: LayoutTensor[dtype, filter_layout, ImmutAnyOrigin],
    c: LayoutTensor[dtype, layout, MutAnyOrigin],
):
    x = Int(block_idx.x * block_dim.x + thread_idx.x)

    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # r = 1, f = 2 * r + 1 = 3

    out_val: Scalar[dtype] = 0
    for r in range(-radius, radius):
        xf = x + r
        if xf >= 0 and xf < M:
            f = filter.load[1](0, r + radius)
            in_val = a.load[1](0, xf)
            out_val += in_val * f

    c.store[1](0, x, out_val)


def equal_1d[
    dtype: DType, a_layout: Layout, b_layout: Layout
](
    out_tensor: LayoutTensor[dtype, b_layout, MutAnyOrigin],
    ref_tensor: LayoutTensor[dtype, a_layout, MutAnyOrigin],
) -> Bool:
    shape = ref_tensor.runtime_layout.shape.value

    for m in range(shape[0]):
        out_val = out_tensor.load[1](0, m)
        ref_val = ref_tensor.load[1](0, m)

        if out_val != ref_val:
            print(0, m)
            print("Out: ", out_val)
            print("Ref: ", ref_val)
            return False
    return True


def main():
    comptime M = 128
    comptime radius = 2
    comptime filter_size = 2 * radius + 1
    comptime BM = 16

    comptime dtype = DType.float32
    var ctx = DeviceContext()

    var a_h = ctx.enqueue_create_host_buffer[dtype](M)
    var filter_h = ctx.enqueue_create_host_buffer[dtype](filter_size)
    var c_h = ctx.enqueue_create_host_buffer[dtype](M)

    var c_ref = ctx.enqueue_create_host_buffer[dtype](M)

    randn[dtype](a_h.unsafe_ptr(), len(a_h))
    randn[dtype](filter_h.unsafe_ptr(), len(filter_h))

    var a_d = ctx.enqueue_create_buffer[dtype](M)
    var filter_d = ctx.enqueue_create_buffer[dtype](filter_size)

    var c_d = ctx.enqueue_create_buffer[dtype](M)

    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(filter_d, filter_h)

    ctx.synchronize()

    comptime layout = Layout.row_major(M)
    comptime filter_layout = Layout.row_major(filter_size)

    a_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](a_d)
    filter_tensor = LayoutTensor[dtype, filter_layout, ImmutAnyOrigin](filter_h)
    c_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](c_d)

    comptime kernel = conv_1d_basic[
        dtype, layout, filter_layout, M, radius, filter_size
    ]

    ctx.enqueue_function_checked[kernel, kernel](
        a_tensor,
        filter_tensor,
        c_tensor,
        grid_dim=(ceildiv(M, BM),),
        block_dim=(BM,),
    )

    a_ref_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](a_h)
    filter_ref_tensor = LayoutTensor[dtype, filter_layout, ImmutAnyOrigin](
        filter_h
    )
    c_ref_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](c_ref)

    conv1d_ref[
        dtype,
        layout,
        filter_layout,
        M,
        radius,
        filter_size,
    ](a_ref_tensor, filter_ref_tensor, c_ref_tensor)

    ctx.enqueue_copy(c_h, c_d)
    ctx.synchronize()

    c_host_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](c_h)

    if equal_1d(c_host_tensor, c_ref_tensor):
        print("Pass")
    else:
        print("Fail")
