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

from random.random import randn

fn row_matmul[a_layout: Layout, b_layout: Layout, c_layout: Layout, M: Int, N: Int, K: Int]():
    pass

def main():
    comptime M = 16
    comptime N = 16
    comptime K = 16
    comptime dtype = DType.float32
    var ctx = DeviceContext()

    var a_h = ctx.enqueue_create_host_buffer[dtype](M * K)
    var b_h = ctx.enqueue_create_host_buffer[dtype](K * N)
    var c_h = ctx.enqueue_create_host_buffer[dtype](M * N)

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

    a_tensor = LayoutTensor[mut=False, dtype, a_layout](a_d)
    b_tensor = LayoutTensor[mut=False, dtype, b_layout](b_d)
    c_tensor = LayoutTensor[mut=True, dtype, c_layout](c_d)

    print_layout(a_layout)