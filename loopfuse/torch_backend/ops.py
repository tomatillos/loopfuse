from enum import Enum
from dataclasses import dataclass
import math
from typing import TypeVar

import torch

from loopfuse import ir


class TransformType(Enum):
    VIEW = "view"
    EXPAND = "expand"
    TRANSPOSE = "transpose"
    CLONE = "clone"


@dataclass
class TensorTransform:
    type: TransformType
    target_shape: tuple[ir.Node | int, ...]
    extra: dict | None = None


@dataclass
class TiledTensor:
    """Allows lazy tilings"""

    torch_tensor: torch.Tensor | None
    tiling: ir.Tiling | None
    name: str
    base_shape: tuple[ir.Node | int, ...]
    transforms: list[TensorTransform]
    paged_metadata: dict | None = None

    @property
    def current_shape(self) -> tuple[ir.Node | int, ...]:
        return self.transforms[-1].target_shape if self.transforms else self.base_shape


@dataclass
class ProgramData:
    """The output data of a program"""

    program: ir.Block
    output: list[TiledTensor]
    internal_tiledtensors: list[TiledTensor]


def create_tiledtensor(tiling: ir.Tiling) -> TiledTensor:
    return TiledTensor(
        torch_tensor=None,
        tiling=tiling,
        name=tiling.name,
        base_shape=tuple(tiling.tensor_dims),
        transforms=[],
    )


def create_loop_vars(tiling: ir.Tiling) -> tuple[ir.Variable, ...]:
    loop_vars = []
    for tile_idx in range(len(tiling.tile_dims)):
        loop_vars.append(ir.Variable(f"index_{tile_idx}_{tiling.dim_names[tile_idx]}"))
    return tuple(loop_vars)


def tiling_like(tiling: ir.Tiling, name: str) -> ir.Tiling:
    return ir.Tiling(
        name=name,
        tensor_dims=tiling.tensor_dims,
        tile_dims=tiling.tile_dims,
        dim_names=tiling.dim_names,
        order=tiling.order,
        dtype=tiling.dtype,
    )


def build_nested_loops(
    loop_vars: tuple[ir.Variable, ...], num_tiles: list, inner_body: ir.Block
) -> ir.Block:
    """Build nested loops from innermost to outermost"""
    program = inner_body
    for dim_idx in reversed(range(len(loop_vars))):
        loop_var = loop_vars[dim_idx]
        program = ir.Block(
            [ir.Loop(loop_var, ir.Constant(0), num_tiles[dim_idx], program)]
        )
    return program


def is_constant_dim(dim: ir.Node) -> bool:
    return (
        isinstance(dim, ir.SymInt) and dim.name.startswith("CONST")
    ) or dim == ir.Constant(1)


def broadcast_load_indices(
    tensor_dims: tuple[ir.Node, ...], loop_vars: tuple[ir.Variable, ...]
) -> tuple[ir.Node, ...]:
    """Broadcasts loads of dimensions that are 1."""
    indices: list[ir.Node] = []
    for i, dim in enumerate(tensor_dims):
        if dim == ir.Constant(1):
            indices.append(ir.Constant(0))
        else:
            indices.append(loop_vars[i])
    return tuple(indices)


T = TypeVar("T")


def permute(lst: list[T], i: int, j: int) -> list[T]:
    lst[i], lst[j] = lst[j], lst[i]
    return lst


def is_single_dim_insertion(from_shape: list[int], to_shape: list[int]) -> bool:
    if len(to_shape) != len(from_shape) + 1:
        return False

    for insert_pos in range(len(to_shape)):
        if to_shape[insert_pos] == 1:
            reconstructed = to_shape[:insert_pos] + to_shape[insert_pos + 1 :]
            if reconstructed == from_shape:
                return True
    return False


def create_load(
    tiledtensor: TiledTensor,
    tup_index: tuple[ir.Node, ...],
    in_matmul: bool = False,
    other: ir.Constant | None = None,
) -> ir.Load:
    assert tiledtensor.tiling is not None
    tiling = tiledtensor.tiling

    if not tiledtensor.transforms:
        return ir.Load(tiledtensor.tiling, ir.Index(tup_index), other)

    tensor_dims = list(tiledtensor.tiling.tensor_dims)
    tile_dims = list(tiledtensor.tiling.tile_dims)
    dim_names = list(tiledtensor.tiling.dim_names)
    order = list(tiledtensor.tiling.order)
    index = list(tup_index)

    for transform_ix in range(len(tiledtensor.transforms) - 1, -1, -1):
        transform = tiledtensor.transforms[transform_ix]
        if (
            in_matmul
            and transform_ix == len(tiledtensor.transforms) - 1
            and transform.type == TransformType.VIEW
        ):
            # skip the last one...
            continue
        if transform.type == TransformType.TRANSPOSE:
            dim0, dim1 = transform.extra["dim0"], transform.extra["dim1"]
            for lst in [tensor_dims, tile_dims, dim_names, order, index]:
                permute(lst, dim0, dim1)

        elif transform.type == TransformType.EXPAND:
            expansion_info = transform.extra["expansion_info"]
            for i, is_expanded in enumerate(expansion_info):
                if is_expanded:
                    tensor_dims[i] = ir.Constant(1)
                    tile_dims[i] = ir.Constant(1)
                    dim_names[i] = f"flat_{dim_names[i]}_ix_{i}"
                    index[i] = ir.Constant(0)
        elif transform.type == TransformType.VIEW:
            from_shape = (
                tiledtensor.transforms[transform_ix - 1].target_shape
                if transform_ix > 0
                else tiledtensor.base_shape
            )
            to_shape = transform.target_shape

            # there are two types of view ops we will handle
            # 1. M,N -> M,1,N (unsqueeze)
            # 2. M,N -> M*N (contract)
            if len(from_shape) + 1 == len(to_shape):
                # check if it's of the case: M,N -> M,1,N
                unsqueeze_case = False
                for insert_ix, d in enumerate(to_shape):
                    if d == 1 or d == ir.SymInt("CONST_1") or d == ir.Constant(1):
                        shape_to_compare = (
                            to_shape[:insert_ix] + to_shape[insert_ix + 1 :]
                        )
                        if shape_to_compare == from_shape:
                            unsqueeze_case = True
                            break
                if unsqueeze_case:
                    tensor_dims.pop(insert_ix)
                    tile_dims.pop(insert_ix)
                    dim_names.pop(insert_ix)
                    index.pop(insert_ix)
                    order_ix_removed = order.pop(insert_ix)
                    order = [
                        o_ix if o_ix < order_ix_removed else o_ix - 1 for o_ix in order
                    ]
                    continue

                # check if it's the case: MN -> M, N
                factor_case = False
                for factor_ix, (from_dim, to_dim) in enumerate(
                    zip(from_shape[:-1], to_shape[:-1])
                ):
                    if from_dim == to_dim:
                        continue
                    elif from_shape[factor_ix + 1 :] == to_shape[factor_ix + 2 :]:
                        factor_case = True
                    break
                if factor_case:
                    N = tensor_dims.pop(factor_ix + 1)
                    M = tensor_dims[factor_ix]
                    tensor_dims[factor_ix] = ir.BinaryOp("*", M, N)

                    n_tile_dim = tile_dims.pop(factor_ix + 1)
                    m_tile_dim = tile_dims.pop(factor_ix)
                    # if m_tile_dim != ir.Constant(1) or n_tile_dim != ir.Constant(1):
                    #     raise NotImplementedError(
                    #         f"View operation not supported: couldn't contract {from_shape} to {to_shape}, only supports trivial tilings on the contracted dimension"
                    #     )
                    tile_dims.insert(
                        factor_ix, ir.BinaryOp("*", m_tile_dim, n_tile_dim)
                    )

                    # index
                    n_index = index.pop(factor_ix + 1)
                    m_index = index.pop(factor_ix)
                    joint_index = ir.BinaryOp(
                        "+", ir.BinaryOp("*", m_index, N), n_index
                    )
                    index.insert(factor_ix, joint_index)

                    # order
                    n_order = order.pop(factor_ix + 1)
                    assert order[factor_ix] == n_order - 1, (
                        "only standard ordering allowed"
                    )
                    order = [o_ix if o_ix < n_order else o_ix - 1 for o_ix in order]

                    n_dim_name = dim_names.pop(factor_ix + 1)
                    m_dim_name = dim_names.pop(factor_ix)
                    dim_names.insert(factor_ix, f"c_{m_dim_name}_ix_{n_dim_name}")

                else:
                    raise NotImplementedError(
                        f"View operation not supported: couldn't contract {from_shape} to {to_shape}"
                    )

            elif len(from_shape) == len(to_shape) + 1:
                # check if it's of the form: M,N -> M*N
                contract_case = False
                for contract_ix, (from_dim, to_dim) in enumerate(
                    zip(from_shape, to_shape)
                ):
                    if from_dim == to_dim:
                        continue
                    elif from_shape[contract_ix + 2 :] == to_shape[contract_ix + 1 :]:
                        contract_case = True
                        break
                if not contract_case:
                    raise NotImplementedError(
                        f"View operation not supported: couldn't contract {from_shape} to {to_shape}"
                    )

                m = from_shape[contract_ix]
                n = from_shape[contract_ix + 1]
                mn_tensor_dim = tensor_dims.pop(contract_ix)
                m_tensor_dim = ir.BinaryOp("//", mn_tensor_dim, n)
                n_tensor_dim = ir.BinaryOp("//", mn_tensor_dim, m)
                tensor_dims.insert(contract_ix, n_tensor_dim)
                tensor_dims.insert(contract_ix, m_tensor_dim)

                mn_tile_dim = tile_dims.pop(contract_ix)
                if mn_tile_dim != ir.Constant(1):
                    raise NotImplementedError(
                        f"Contract operation not supported: couldn't contract {from_shape} to {to_shape}, only supports trivial tilings on the contracted dimension"
                    )
                m_tile_dim = ir.Constant(1)
                n_tile_dim = ir.Constant(1)
                tile_dims.insert(contract_ix, n_tile_dim)
                tile_dims.insert(contract_ix, m_tile_dim)

                # index
                mn_index = index.pop(contract_ix)
                m_index = ir.BinaryOp("//", mn_index, n)
                n_index = ir.BinaryOp("%", mn_index, n)
                index.insert(contract_ix, n_index)
                index.insert(contract_ix, m_index)

                # order
                mn_order = order.pop(contract_ix)
                order = [o_ix if o_ix < mn_order else o_ix + 1 for o_ix in order]
                order.insert(contract_ix, mn_order + 1)
                order.insert(contract_ix, mn_order)

                # dim names
                mn_dim_name = dim_names.pop(contract_ix)
                m_dim_name = f"c_{mn_dim_name}_ix_m"
                n_dim_name = f"c_{mn_dim_name}_ix_n"
                dim_names.insert(contract_ix, n_dim_name)
                dim_names.insert(contract_ix, m_dim_name)
            elif (
                len(from_shape) == len(to_shape)
                and sum(1 for f, t in zip(from_shape, to_shape) if f != t) <= 1
            ):
                # no op - shapes are actually the same, but with different symint labels.
                pass
            else:
                raise NotImplementedError(
                    f"View operation not supported: couldn't contract {from_shape} to {to_shape}"
                )

        elif transform.type == TransformType.CLONE:
            pass
        else:
            raise NotImplementedError(
                "View operation not supported: complex dimension insertion"
            )

    tiling = ir.Tiling(
        name=f"{tiledtensor.name}",
        tensor_dims=tuple(tensor_dims),
        tile_dims=tuple(tile_dims),
        dim_names=tuple(dim_names),
        order=tuple(order),
        dtype=tiledtensor.tiling.dtype,
        paged_metadata=tiledtensor.paged_metadata,
    )

    return ir.Load(tiling, ir.Index(tuple(index)), other)


def build_matmul(
    output_name: str,
    A: TiledTensor,
    B: TiledTensor,
) -> ProgramData:
    assert A.tiling is not None
    assert B.tiling is not None
    assert len(A.tiling.tensor_dims) == len(B.tiling.tensor_dims)
    # assert A.tiling.tensor_dims[:-2] == B.tiling.tensor_dims[:-2]
    # assert A.tiling.tensor_dims[-1] == B.tiling.tensor_dims[-2]
    # assert A.tiling.dtype == B.tiling.dtype, f"{A.tiling.dtype} != {B.tiling.dtype}"

    output_tiling = ir.Tiling(
        name=output_name,
        tensor_dims=A.tiling.tensor_dims[:-1] + B.tiling.tensor_dims[-1:],
        tile_dims=A.tiling.tile_dims[:-1] + B.tiling.tile_dims[-1:],
        dim_names=A.tiling.dim_names[:-1] + B.tiling.dim_names[-1:],
        order=tuple(range(len(A.tiling.tensor_dims))),
        dtype=A.tiling.dtype,
    )
    output_tiledtensor = create_tiledtensor(output_tiling)
    output_loop_vars = create_loop_vars(output_tiling)

    batch_vars = output_loop_vars[:-2]
    m_var = output_loop_vars[-2]
    n_var = output_loop_vars[-1]
    k_is_constant = is_constant_dim(A.tiling.tile_dims[-1])
    k_var: ir.Node
    if k_is_constant:
        k_var = ir.Constant(0)
    else:
        k_var = ir.Variable(
            f"{A.tiling.dim_names[-1]}_ix_{len(A.tiling.tile_dims) - 1}"
        )

    a_indices = batch_vars + (m_var, k_var)
    b_indices = batch_vars + (k_var, n_var)
    c_indices = batch_vars + (m_var, n_var)

    a_tile = create_load(A, a_indices, in_matmul=True, other=ir.Constant(0))
    b_tile = create_load(B, b_indices, in_matmul=True, other=ir.Constant(0))

    ab_product = ir.BinaryOp("@", a_tile, b_tile)

    # todo: clean this logic up
    if k_is_constant:
        ab_to_dtype = ir.Cast(ab_product, A.tiling.dtype)
        store_result = ir.Store(output_tiling, ir.Index(c_indices), ab_to_dtype)
        inner_body = ir.Block([store_result])
        n_loop = ir.Loop(n_var, ir.Constant(0), B.tiling.num_tiles[-1], inner_body)
        n_loop_load_a = ir.Block([a_tile, n_loop])
        program = build_nested_loops(
            output_loop_vars[:-1], list(output_tiling.num_tiles), n_loop_load_a
        )
    else:
        acc = ir.Variable("matmul_acc")
        acc_shape = (A.tiling.tile_dims[-2], B.tiling.tile_dims[-1])
        init_acc = ir.Assign(acc, ir.Zeros(acc_shape, "float32"))
        update_acc = ir.Assign(acc, ir.BinaryOp("+", acc, ab_product))
        acc_to_dtype = ir.Cast(acc, A.tiling.dtype)
        store_result = ir.Store(output_tiling, ir.Index(c_indices), acc_to_dtype)
        k_loop_body = ir.Block([update_acc])
        k_loop = ir.Loop(k_var, ir.Constant(0), A.tiling.num_tiles[-1], k_loop_body)
        inner_body = ir.Block([init_acc, k_loop, store_result])

        program = build_nested_loops(
            output_loop_vars, list(output_tiling.num_tiles), inner_body
        )

    return ProgramData(program, [output_tiledtensor], [])


def build_softmax(
    output_name: str,
    X: TiledTensor,
    dim: int,
    half_to_float: bool = False,
) -> ProgramData:
    upcast_X = build_cast(f"{X.name}_upcast", X, "float32")
    X_log2e = build_binary_op(
        f"{X.name}_scaled", upcast_X.output[0], ir.Constant(1 / math.log(2)), "*"
    )
    scan_max = build_scan_max(f"{X.name}_scan_max", X_log2e.output[0])
    m_ik = scan_max.output[0]
    m_i = scan_max.output[1]

    safe_exp_program = build_safe_exp(X_log2e.output[0], m_ik, m_i)
    safe_exp_x = safe_exp_program.output[0]

    sum_program = build_reduce(f"sum_{safe_exp_x.name}", safe_exp_x, "sum")
    sum_safe_exp_x = sum_program.output[0]

    softmax_program = build_binary_op(
        "softmax_x", safe_exp_x, sum_safe_exp_x, "/", output_dtype=X.tiling.dtype
    )
    softmax_x = softmax_program.output[0]

    internal_tensors = [m_ik, m_i, safe_exp_x, sum_safe_exp_x]
    combined_program = ir.Block(
        [
            upcast_X.program,
            X_log2e.program,
            scan_max.program,
            safe_exp_program.program,
            sum_program.program,
            softmax_program.program,
        ]
    )
    return ProgramData(combined_program, [softmax_x], internal_tensors)


def build_scan_max(output_name: str, X: TiledTensor) -> ProgramData:
    X_tiling = X.tiling
    assert X_tiling is not None
    loop_vars = create_loop_vars(X_tiling)

    k_var = loop_vars[-1]
    num_k_tiles = X_tiling.num_tiles[-1]

    m_ik_tiling = ir.Tiling(
        f"{X_tiling.name}_max",
        X_tiling.tensor_dims[:-1] + (num_k_tiles,),
        X_tiling.tile_dims[:-1] + (ir.Constant(1),),
        X_tiling.dim_names[:-1]
        + (f"{X_tiling.dim_names[-1]}_ix_{len(X_tiling.tile_dims) - 1}",),
        tuple(range(len(X_tiling.tensor_dims))),
        "float32",
    )
    m_ik_tiledtensor = create_tiledtensor(m_ik_tiling)

    mi_tiling = ir.Tiling(
        f"{X_tiling.name}_mi",
        X_tiling.tensor_dims[:-1] + (ir.Constant(1),),
        X_tiling.tile_dims[:-1] + (ir.Constant(1),),
        X_tiling.dim_names[:-1]
        + (f"{X_tiling.dim_names[-1]}_ix_{len(X_tiling.tile_dims) - 1}",),
        tuple(range(len(X_tiling.tensor_dims))),
        "float32",
    )
    mi_tiledtensor = create_tiledtensor(mi_tiling)

    mi_tile = ir.Variable("running_max")
    init_mi = ir.Assign(
        mi_tile,
        ir.Full(mi_tiling.tile_dims[-2:], ir.Constant(-float("inf")), "float32"),
    )

    x_tile = create_load(X, loop_vars, other=ir.Constant(-float("inf")))
    max_reduced = ir.Reduce("max", x_tile, -1)
    new_max = ir.BinaryOp("max", mi_tile, max_reduced)

    store_m_ik = ir.Store(m_ik_tiling, ir.Index(loop_vars), new_max)
    update_mi = ir.Assign(mi_tile, new_max)
    store_mi = ir.Store(
        mi_tiling, ir.Index(loop_vars[:-1] + (ir.Constant(0),)), mi_tile
    )

    k_loop_body = ir.Block([store_m_ik, update_mi])
    k_loop = ir.Loop(k_var, ir.Constant(0), num_k_tiles, k_loop_body)

    inner_body = ir.Block([init_mi, k_loop, store_mi])
    program = build_nested_loops(
        loop_vars[:-1], list(X_tiling.num_tiles[:-1]), inner_body
    )

    return ProgramData(program, [m_ik_tiledtensor, mi_tiledtensor], [])


def build_safe_exp(X: TiledTensor, m_ik: TiledTensor, mi: TiledTensor) -> ProgramData:
    """computes S = exp(X - m_ik) * exp(m_ik - mi)"""
    safe_X_prg = build_binary_op("safe_X", X, m_ik, "-")
    safe_exp_X_prg = build_unary_op("safe_exp_X", safe_X_prg.output[0], "exp2")
    delta_m_prg = build_binary_op("delta_m", m_ik, mi, "-")
    alpha_prg = build_unary_op("alpha", delta_m_prg.output[0], "exp2")
    S_prg = build_binary_op(
        "S",
        safe_exp_X_prg.output[0],
        alpha_prg.output[0],
        "*",
    )
    combined_program = ir.Block(
        [
            safe_X_prg.program,
            safe_exp_X_prg.program,
            delta_m_prg.program,
            alpha_prg.program,
            S_prg.program,
        ]
    )
    return ProgramData(combined_program, S_prg.output, [])


def build_reduce(output_name: str, X: TiledTensor, reduce_op: str) -> ProgramData:
    X_tiling = X.tiling
    assert X_tiling is not None
    loop_vars = create_loop_vars(X_tiling)

    output_tiling = ir.Tiling(
        f"{X_tiling.name}_{reduce_op}_reduced",
        X_tiling.tensor_dims[:-1] + (ir.Constant(1),),
        X_tiling.tile_dims[:-1] + (ir.Constant(1),),
        X_tiling.dim_names[:-1] + (f"{X_tiling.dim_names[-1]}_reduced",),
        tuple(range(len(X_tiling.tensor_dims))),
        X_tiling.dtype,
    )
    output_tiledtensor = create_tiledtensor(output_tiling)

    acc_shape = output_tiling.tile_dims[-2:]
    initial_value: ir.Node
    if reduce_op == "sum":
        initial_value = ir.Zeros(acc_shape, "float32")
        update_op = "+"
        other = ir.Constant(0)
    elif reduce_op == "max":
        initial_value = ir.Full(acc_shape, ir.Constant(-float("inf")), "float32")
        update_op = "max"
        other = ir.Constant(-float("inf"))
    elif reduce_op == "min":
        initial_value = ir.Full(acc_shape, ir.Constant(float("inf")), "float32")
        update_op = "min"
        other = ir.Constant(float("inf"))
    else:
        raise ValueError(f"Unsupported reduction operation: {reduce_op}")

    acc = ir.Variable(f"{reduce_op}_acc")
    x_tile = create_load(X, loop_vars, other=other)
    reduced_tile = ir.Reduce(reduce_op, x_tile, -1)

    init_acc = ir.Assign(acc, initial_value)
    update_acc = ir.Assign(acc, ir.BinaryOp(update_op, acc, reduced_tile))
    store_result = ir.Store(
        output_tiling, ir.Index(loop_vars[:-1] + (ir.Constant(0),)), acc
    )

    reduction_loop = ir.Loop(
        loop_vars[-1], ir.Constant(0), X_tiling.num_tiles[-1], ir.Block([update_acc])
    )
    inner_body = ir.Block([init_acc, reduction_loop, store_result])
    program = build_nested_loops(
        loop_vars[:-1], list(X_tiling.num_tiles[:-1]), inner_body
    )

    return ProgramData(program, [output_tiledtensor], [])


def build_unary_op(output_name: str, X: TiledTensor, op: str) -> ProgramData:
    X_tiling = X.tiling
    assert X_tiling is not None

    output_tiling = tiling_like(X_tiling, output_name)
    output_tiledtensor = create_tiledtensor(output_tiling)
    loop_vars = create_loop_vars(X_tiling)

    x_tile = create_load(X, loop_vars)
    result = ir.UnaryOp(op, x_tile)
    store_result = ir.Store(output_tiling, ir.Index(loop_vars), result)

    program = build_nested_loops(
        loop_vars, list(X_tiling.num_tiles), ir.Block([store_result])
    )
    return ProgramData(program, [output_tiledtensor], [])


def build_binary_op(
    output_name: str,
    X: TiledTensor,
    Y: TiledTensor | ir.Constant,
    op: str,
    output_dtype: str | None = None,
) -> ProgramData:
    X_tiling = X.tiling
    assert X_tiling is not None

    output_dtype_specified = output_dtype is not None
    if not output_dtype_specified:
        output_dtype = X_tiling.dtype

    output_tiling = ir.Tiling(
        name=output_name,
        tensor_dims=X_tiling.tensor_dims,
        tile_dims=X_tiling.tile_dims,
        dim_names=X_tiling.dim_names,
        order=tuple(range(len(X_tiling.tensor_dims))),
        dtype=output_dtype,
    )
    output_tiledtensor = create_tiledtensor(output_tiling)
    loop_vars = create_loop_vars(X_tiling)

    x_tile = create_load(X, loop_vars)

    y_tile: ir.Node
    if isinstance(Y, TiledTensor):
        assert Y.tiling is not None
        y_indices = broadcast_load_indices(Y.tiling.tensor_dims, loop_vars)
        y_tile = create_load(Y, y_indices)
    elif isinstance(Y, ir.Constant):
        y_tile = Y
    elif isinstance(Y, ir.SymInt):
        y_tile = Y
    else:
        raise ValueError(f"Unsupported type: {type(Y)}")

    result = ir.BinaryOp(op, x_tile, y_tile)
    if output_dtype_specified:
        result = ir.Cast(result, output_dtype)
    store_result = ir.Store(output_tiling, ir.Index(loop_vars), result)

    program = build_nested_loops(
        loop_vars, list(X_tiling.num_tiles), ir.Block([store_result])
    )
    return ProgramData(program, [output_tiledtensor], [])


def build_cast(output_name: str, X: TiledTensor, dtype: str) -> ProgramData:
    X_tiling = X.tiling
    assert X_tiling is not None

    output_tiling = tiling_like(X_tiling, output_name)
    output_tiledtensor = create_tiledtensor(output_tiling)
    loop_vars = create_loop_vars(X_tiling)

    x_tile = create_load(X, loop_vars)
    cast_x_tile = ir.Cast(x_tile, dtype)
    store_result = ir.Store(output_tiling, ir.Index(loop_vars), cast_x_tile)

    program = build_nested_loops(
        loop_vars, list(X_tiling.num_tiles), ir.Block([store_result])
    )
    return ProgramData(program, [output_tiledtensor], [])


def build_causal_mask(output_name: str, A: TiledTensor) -> ProgramData:
    # For tensor of shape [.., M, N]
    # Masks the upper triangle of the last two dimensions
    assert A.tiling is not None

    output_tiling = tiling_like(A.tiling, output_name)
    output_tiledtensor = create_tiledtensor(output_tiling)
    loop_vars = create_loop_vars(A.tiling)

    tM = A.tiling.tile_dims[-2]
    tN = A.tiling.tile_dims[-1]

    arange_m = ir.Arange(ir.Constant(0), tM)
    arange_n = ir.Arange(ir.Constant(0), tN)
    offs_m = ir.BinaryOp("*", loop_vars[-2], tM)
    pos_m = ir.BinaryOp("+", offs_m, arange_m)
    offs_n = ir.BinaryOp("*", loop_vars[-1], tN)
    pos_n = ir.BinaryOp("+", offs_n, arange_n)

    pos_m_unsqueezed = ir.Unsqueeze(pos_m, 1)
    pos_n_unsqueezed = ir.Unsqueeze(pos_n, 0)
    mask = ir.BinaryOp(">=", pos_m_unsqueezed, pos_n_unsqueezed, name="causal_mask")
    a_tile = create_load(A, loop_vars)
    masked_a = ir.Where(mask, a_tile, ir.Constant(-float("inf")), name="masked_block")

    store_result = ir.Store(output_tiling, ir.Index(loop_vars), masked_a)

    program = build_nested_loops(
        loop_vars, list(A.tiling.num_tiles), ir.Block([store_result])
    )
    return ProgramData(program, [output_tiledtensor], [])


def build_sliding_mask(
    output_name: str, A: TiledTensor, window_size: ir.Constant
) -> ProgramData:
    # For tensor of shape [.., M, N]
    # Masks positions outside the sliding window around each position
    assert A.tiling is not None
    assert isinstance(window_size, (ir.Constant, ir.SymInt, int))

    output_tiling = tiling_like(A.tiling, output_name)
    output_tiledtensor = create_tiledtensor(output_tiling)
    loop_vars = create_loop_vars(A.tiling)

    tM = A.tiling.tile_dims[-2]
    tN = A.tiling.tile_dims[-1]

    arange_m = ir.Arange(ir.Constant(0), tM)
    arange_n = ir.Arange(ir.Constant(0), tN)
    offs_m = ir.BinaryOp("*", loop_vars[-2], tM)
    pos_m = ir.BinaryOp("+", offs_m, arange_m)
    offs_n = ir.BinaryOp("*", loop_vars[-1], tN)
    pos_n = ir.BinaryOp("+", offs_n, arange_n)

    pos_m_unsqueezed = ir.Unsqueeze(pos_m, 1)
    pos_n_unsqueezed = ir.Unsqueeze(pos_n, 0)

    # Create sliding window mask: pos_m - pos_n <= window_size
    diff = ir.BinaryOp("-", pos_m_unsqueezed, pos_n_unsqueezed)

    sliding_mask = ir.BinaryOp(
        "<=", diff, ir.BinaryOp("-", window_size, ir.Constant(1))
    )
    causal_mask = ir.BinaryOp(">=", diff, ir.Constant(0))

    mask = ir.BinaryOp("&", sliding_mask, causal_mask)

    a_tile = create_load(A, loop_vars)
    masked_a = ir.Where(mask, a_tile, ir.Constant(-float("inf")))

    store_result = ir.Store(output_tiling, ir.Index(loop_vars), masked_a)

    program = build_nested_loops(
        loop_vars, list(A.tiling.num_tiles), ir.Block([store_result])
    )
    return ProgramData(program, [output_tiledtensor], [])
