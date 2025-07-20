from __future__ import annotations
import dataclasses
import sys
import tempfile
import operator
import os
import logging
from typing import Callable, Any
from functools import partial
import importlib.util

import torch
from torch._functorch.aot_autograd import aot_module_simplified
from torch._dynamo.source import LocalSource, TensorPropertySource, Source

from loopfuse import ir, optimize
from loopfuse.compiler.passes import cleanup_passes
from loopfuse.torch_backend import ops
from loopfuse.torch_backend.ops import TiledTensor, TensorTransform, TransformType
from loopfuse.codegen import KernelData, codegen_program


DEBUG = os.environ.get("DEBUG", "0") == "1"
USE_HARDCODE_KERNEL = os.environ.get("USE_HARDCODE_KERNEL", "0") == "1"
CODEGEN_ONLY = os.environ.get("CODEGEN_ONLY", "0") == "1"
try:
    import triton
except ImportError:
    CODEGEN_ONLY = True


def get_base_name_from_source(source: Source) -> str | None:
    if isinstance(source, LocalSource):
        return source.local_name
    if hasattr(source, "base"):
        return get_base_name_from_source(source.base)
    return None


def get_node_name(node: torch.fx.Node) -> str:
    if hasattr(node, "_dynamo_source"):
        if isinstance(node._dynamo_source, LocalSource):  # type: ignore
            return node._dynamo_source.local_name  # type: ignore
    if node.name.startswith("l_"):
        return node.name[2:-1]
    return node.name


def get_input_names(gm: torch.fx.GraphModule) -> list[str]:
    input_names = []
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            input_names.append(get_node_name(n))
        else:
            pass
    return input_names


def get_symint_names(gm: torch.fx.GraphModule) -> dict[str, str]:
    symint_names = {}
    # todo: flatten this monstrosity
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            if hasattr(n, "_dynamo_source"):
                source = n._dynamo_source
                if isinstance(source, TensorPropertySource):
                    symint_name_in_graph = n.name
                    descriptive_name = symint_name_in_graph
                    base_name = get_base_name_from_source(source.base)
                    if base_name:
                        dim_idx = source.idx
                        descriptive_name = f"{base_name}_dim{dim_idx}"
                    symint_names[symint_name_in_graph] = descriptive_name
    return symint_names


def loopfuse_backend(gm, sample_inputs, mode="max-autotune", **kwargs):
    """Main entry point for the loopfuse torch.compile backend."""
    input_names = get_input_names(gm)
    symint_names = get_symint_names(gm)
    if mode:
        kwargs["mode"] = mode
    compiler = partial(
        loopfuse_compiler, input_names=input_names, symint_names=symint_names, **kwargs
    )
    return aot_module_simplified(gm, sample_inputs, fw_compiler=compiler)


def sum_helper(output_name, X: TiledTensor, dim: list[ir.Constant], keepdim: bool):
    if len(dim) != 1 or not isinstance(dim[0], ir.Constant) or not dim[0].value == -1:
        raise NotImplementedError("sum with dim != -1 not supported")
    elif not keepdim:
        raise NotImplementedError("sum with keepdim=False not supported")
    return ops.build_reduce(output_name, X, "sum")


def max_helper(
    output_name, X: TiledTensor, dim: list[ir.Constant], keepdim: bool
) -> ir.ProgramData:
    if not isinstance(dim, ir.Constant) or not dim.value == -1:
        raise NotImplementedError("max with dim != -1 not supported")
    elif not keepdim:
        raise NotImplementedError("max with keepdim=False not supported")
    program_data = ops.build_reduce(output_name, X, "max")
    program_data.output.append(None)  # type: ignore
    return program_data


def to_helper(output_name, X: TiledTensor, dtype: torch.dtype, *args, **kwargs):
    dtype_str = str(dtype).split("torch.")[-1]
    return ops.build_cast(output_name, X, dtype_str)


def pow_helper(output_name: str, X: TiledTensor, exponent: ir.Constant):
    if exponent.value == 2:
        return ops.build_unary_op(output_name, X, "square")
    raise NotImplementedError(f"pow with exponent={exponent} not supported.")


torch_func_map = {
    torch.ops.aten.mm.default: ops.build_matmul,
    torch.ops.aten.bmm.default: ops.build_matmul,
    torch.ops.aten.add.Tensor: partial(ops.build_binary_op, op="+"),
    torch.ops.aten.sub.Tensor: partial(ops.build_binary_op, op="-"),
    torch.ops.aten.mul.Tensor: partial(ops.build_binary_op, op="*"),
    torch.ops.aten.div.Tensor: partial(ops.build_binary_op, op="/"),
    torch.ops.aten.neg.default: partial(ops.build_binary_op, op="*", Y=ir.Constant(-1)),
    torch.ops.aten.exp.default: partial(ops.build_unary_op, op="exp"),
    torch.ops.aten.exp2.default: partial(ops.build_unary_op, op="exp2"),
    torch.ops.aten.tanh.default: partial(ops.build_unary_op, op="tanh"),
    torch.ops.aten.sin.default: partial(ops.build_unary_op, op="sin"),
    torch.ops.aten.cos.default: partial(ops.build_unary_op, op="cos"),
    torch.ops.aten.relu.default: partial(ops.build_unary_op, op="relu"),
    torch.ops.aten._softmax.default: ops.build_softmax,
    torch.ops.loopfuse.scan_max.default: ops.build_scan_max,
    torch.ops.loopfuse.causal_mask.default: ops.build_causal_mask,
    torch.ops.loopfuse.sliding_mask.default: ops.build_sliding_mask,
    torch.ops.aten.sum.dim_IntList: sum_helper,
    torch.ops.aten.max.dim: max_helper,
    torch.ops.aten._to_copy.default: to_helper,
    torch.ops.aten.sum.default: partial(ops.build_reduce, reduce_op="sum"),
    torch.ops.aten.max.default: partial(ops.build_reduce, reduce_op="max"),
    torch.ops.aten.min.default: partial(ops.build_reduce, reduce_op="min"),
    torch.ops.aten.pow.Tensor_Scalar: pow_helper,
}

inplace_unary_ops = {
    torch.ops.aten.exp.default: "exp",
    torch.ops.aten.exp2.default: "exp2",
    torch.ops.aten.log.default: "log",
    torch.ops.aten.tanh.default: "tanh",
    torch.ops.aten.sin.default: "sin",
    torch.ops.aten.cos.default: "cos",
    torch.ops.aten.relu.default: "relu",
}

def lookup_torch_func(_func):
    if _func in torch_func_map:
        return torch_func_map[_func]
    else:
        raise ValueError(f"No function found for {_func}")


def loopfuse_compiler(
    gm: torch.fx.GraphModule,
    sample_inputs: list[torch.Tensor],
    input_names: list[str] | None = None,
    symint_names: dict[str, str] | None = None,
    mode: str | None = None,
) -> Callable:
    """Parses a torch.fx.GraphModule into loopfuse IR, optimizes it, and codegens a Triton kernel."""

    class FunctionCallVisitor:
        def __init__(self, parser: GraphParser):
            self.parser = parser
            self.handlers = {
                operator.floordiv: self.handle_simple_op,
                operator.getitem: self.handle_simple_op,
                operator.mul: self.handle_mul,
                torch.ops.aten.expand.default: self.handle_expand,
                torch.ops.aten.view.default: self.handle_view,
                torch.ops.aten._unsafe_view.default: self.handle_view,
                torch.ops.aten.unsqueeze.default: self.handle_unsqueeze,
                torch.ops.aten.transpose.int: self.handle_transpose,
                torch.ops.aten.clone.default: self.handle_clone,
                torch.ops.aten.sym_size.int: self.handle_sym_size,
            }

        def visit(self, node: torch.fx.Node):
            _args = self.parser._load_arg(node.args)
            _kwargs = self.parser._load_arg(node.kwargs)
            _args = self.parser._rewrite_args_to_symint(_args)

            handler = self.handlers.get(node.target)
            if handler:
                handler(node, _args, _kwargs)
            else:
                self.handle_default(node, _args, _kwargs)

        def handle_simple_op(self, node: torch.fx.Node, _args, _kwargs):
            if callable(node.target):
                self.parser.env[node.name] = node.target(*_args, **_kwargs)
            else:
                raise TypeError(f"node.target '{node.target}' is not callable")

        def handle_mul(self, node: torch.fx.Node, _args, _kwargs):
            if isinstance(_args[0], ir.BinaryOp):
                self.parser.env[node.name] = ir.BinaryOp("*", _args[0], _args[1])
            else:
                self.handle_simple_op(node, _args, _kwargs)

        def handle_expand(self, node: torch.fx.Node, _args, _kwargs):
            _tiling, new_shape = _args[0], _args[1]
            assert isinstance(_tiling, TiledTensor)
            current_shape = _tiling.current_shape
            if current_shape == tuple(new_shape):
                self.parser.env[node.name] = _tiling
            else:
                view_shape = [
                    current_shape[i] if new_shape[i] == -1 else new_shape[i]
                    for i in range(len(new_shape))
                ]
                expansion_info = [
                    (orig_dim == ir.Constant(1) or orig_dim == ir.SymInt("CONST_1"))
                    for orig_dim in current_shape
                ]
                transform = TensorTransform(
                    TransformType.EXPAND,
                    tuple(view_shape),
                    extra={"expansion_info": expansion_info},
                )
                self.parser.env[node.name] = self.parser.create_transformed_tiling(
                    _tiling, transform
                )

        def handle_view(self, node: torch.fx.Node, _args, _kwargs):
            _tiling, _shape = _args[0], tuple(_args[1])
            assert isinstance(_tiling, TiledTensor)
            if _shape == _tiling.current_shape:
                self.parser.env[node.name] = _tiling
            else:
                transform = TensorTransform(TransformType.VIEW, _shape)
                self.parser.env[node.name] = self.parser.create_transformed_tiling(
                    _tiling, transform
                )

        def handle_unsqueeze(self, node: torch.fx.Node, _args, _kwargs):
            _tiling, _dim = _args
            assert isinstance(_tiling, TiledTensor)
            old_shape = _tiling.current_shape
            new_shape = old_shape[:_dim] + (ir.Constant(1),) + old_shape[_dim:]
            transform = TensorTransform(TransformType.VIEW, new_shape)
            self.parser.env[node.name] = self.parser.create_transformed_tiling(
                _tiling, transform
            )

        def handle_transpose(self, node: torch.fx.Node, _args, _kwargs):
            _tiling, _dim0, _dim1 = _args
            # todo: remove this once we handle constant symints properly
            if isinstance(_dim0, ir.SymInt) and _dim0.name == "CONST_1":
                _dim0 = 1
            assert isinstance(_tiling, TiledTensor)
            shape = list(_tiling.current_shape)
            shape[_dim0], shape[_dim1] = shape[_dim1], shape[_dim0]
            transform = TensorTransform(
                TransformType.TRANSPOSE,
                tuple(shape),
                extra={"dim0": _dim0, "dim1": _dim1},
            )
            self.parser.env[node.name] = self.parser.create_transformed_tiling(
                _tiling, transform
            )

        def handle_clone(self, node: torch.fx.Node, _args, _kwargs):
            _tiling = _args[0]
            assert isinstance(_tiling, TiledTensor)
            transform = TensorTransform(TransformType.CLONE, _tiling.current_shape)
            self.parser.env[node.name] = self.parser.create_transformed_tiling(
                _tiling, transform
            )

        def handle_sym_size(self, node: torch.fx.Node, _args, _kwargs):
            _tiling, _dim = _args
            assert isinstance(_tiling, TiledTensor)
            shape = _tiling.current_shape
            self.parser.env[node.name] = shape[_dim]

        def handle_default(self, node: torch.fx.Node, _args, _kwargs):
            # TODO: add way more things to the inline op stack
            if node.target in inplace_unary_ops:
                transform = TensorTransform(TransformType.UNARY_OP, _args[0].current_shape, extra={"op": inplace_unary_ops[node.target]})
                self.parser.env[node.name] = self.parser.create_transformed_tiling(_args[0], transform)
                return

            create_ir_func = lookup_torch_func(node.target)
            output_name = f"output_{self.parser.tmpvar_counter}"
            self.parser.tmpvar_counter += 1
            # torch does annoying views before bmms, hack to ignore the last viewop
            ignore_last_viewop = node.target == torch.ops.aten.bmm.default
            for arg in _args:
                if isinstance(arg, TiledTensor) and arg.tiling is None:
                    self.parser.init_tiling(arg, ignore_last_viewop)
            _args = self.parser._rewrite_args_to_constants(_args)
            program_data = create_ir_func(output_name, *_args, **_kwargs)
            self.parser.programs = self.parser.programs + program_data.program
            self.parser.created_tilings.extend(program_data.output)
            if len(program_data.output) == 1:
                output = program_data.output[0]
            else:
                output = program_data.output
            self.parser.env[node.name] = output

    class DimensionManager:
        def __init__(self):
            self.torch_symint_map: dict[str, ir.SymInt] = {}
            self.const_int_to_ir: dict[int, ir.SymInt] = {}
            self.tensor_dim_to_tile_dim: dict[ir.SymInt, ir.SymInt] = {}
            self.new_dims_to_create: dict = {}
            self.dim_counter = 0

        def get_new_dim_name(self):
            self.dim_counter += 1
            return f"dim_{self.dim_counter}"

        def from_torch_dim(self, dim) -> ir.SymInt:
            if isinstance(dim, torch.SymInt):
                name = dim.node._graph_repr()
                if name not in self.torch_symint_map:
                    self.torch_symint_map[name] = ir.SymInt(name)
                return self.torch_symint_map[name]
            elif isinstance(dim, int):
                if dim not in self.const_int_to_ir:
                    if dim == -1:
                        self.const_int_to_ir[dim] = ir.SymInt("CONST_NEG_1")
                    else:
                        self.const_int_to_ir[dim] = ir.SymInt(f"CONST_{dim}")
                return self.const_int_to_ir[dim]
            elif isinstance(dim, ir.SymInt):
                return dim
            elif isinstance(dim, ir.BinaryOp):
                if dim in self.new_dims_to_create:
                    return self.new_dims_to_create[dim]
                new_dim = ir.SymInt(self.get_new_dim_name())
                self.new_dims_to_create[dim] = new_dim
                return new_dim
            raise TypeError(f"Unknown dimension type: {type(dim)}")

        def get_tile_dim(self, tensor_dim: ir.SymInt) -> ir.SymInt:
            if tensor_dim not in self.tensor_dim_to_tile_dim:
                if tensor_dim.name.startswith("CONST_"):
                    tile_dim = tensor_dim
                else:
                    tile_dim = ir.SymInt(f"tile_{tensor_dim.name}")
                self.tensor_dim_to_tile_dim[tensor_dim] = tile_dim
            return self.tensor_dim_to_tile_dim[tensor_dim]

    class GraphParser:
        def __init__(
            self,
            sample_inputs: list[torch.Tensor],
            input_names: list[str] | None = None,
            symint_names: dict[str, str] | None = None,
        ):
            self.args_iter = iter(sample_inputs)
            self.env: dict[str, Any] = {}
            self.dim_manager = DimensionManager()
            self.visitor = FunctionCallVisitor(self)

            self.programs = ir.Block([])
            self.graph_inputs: list[TiledTensor | ir.SymInt] = []
            self.graph_tile_dims: set[ir.SymInt] = set()
            self.created_tilings: list[TiledTensor] = []
            self.output_node: Any = None

            self.named_vars: dict[str, int] = {}
            self.tmpvar_counter = 0
            self.input_names = input_names
            self.placeholder_count = 0
            self.symint_names = symint_names if symint_names is not None else {}

        def get_tmp_var(self, name: str) -> str:
            if name not in self.named_vars:
                self.named_vars[name] = self.tmpvar_counter
                self.tmpvar_counter += 1
            return f"tmp_{self.named_vars[name]}"

        def parse_graph(self, gm: torch.fx.GraphModule):
            for node in gm.graph.nodes:
                logging.debug(f"parsing node: {node.format_node()}")
                if node.op == "placeholder":
                    self._parse_placeholder(node)
                elif node.op == "call_function":
                    if node.target == torch.ops.loopfuse.paged_load.default:
                        self._parse_paged_load(node)
                    else:
                        self._parse_call_function(node)
                elif node.op == "output":
                    self._parse_output(node)
                else:
                    raise ValueError(f"Unknown node type: {node.op}")

        def _load_arg(self, arg):
            return torch.fx.graph.map_arg(arg, lambda n: self.env[n.name])

        def _rewrite_args_to_symint(self, args: Any) -> Any:
            if isinstance(args, (list, tuple)):
                return type(args)([self._rewrite_args_to_symint(a) for a in args])
            if isinstance(args, int) and args in self.dim_manager.const_int_to_ir:
                return self.dim_manager.const_int_to_ir[args]
            return args

        def _rewrite_args_to_constants(self, args: Any) -> Any:
            if isinstance(args, bool):
                return args
            if isinstance(args, (list, tuple)):
                return type(args)([self._rewrite_args_to_constants(a) for a in args])
            if isinstance(args, (float, int)):
                return ir.Constant(args)
            return args

        def _create_tensor_and_tile_dims(
            self, shape: tuple[ir.Node | int, ...]
        ) -> tuple[list[ir.SymInt], list[ir.SymInt], list[str]]:
            tensor_dims = [self.dim_manager.from_torch_dim(d) for d in shape]
            tile_dims = [self.dim_manager.get_tile_dim(d) for d in tensor_dims]
            dim_names = [d.name for d in tensor_dims]
            self.graph_tile_dims.update(tile_dims)
            return tensor_dims, tile_dims, dim_names

        def init_tiling(self, arg: TiledTensor, ignore_last_viewop=False):
            """Inplace inits tiling for a TiledTensor"""
            assert arg.tiling is None, f"Tiling already initialized for {arg.name}"
            if ignore_last_viewop and len(arg.transforms) > 0:
                if arg.transforms[-1].type != TransformType.VIEW:
                    raise ValueError(f"Cannot ignore last viewop for {arg.name}")
                arg.transforms = arg.transforms[:-1]
            shape = arg.current_shape
            (
                tensor_dims,
                tile_dims,
                dim_names,
            ) = self._create_tensor_and_tile_dims(shape)
            tile_dims[:-2] = [ir.Constant(1) for _ in tile_dims[:-2]]  # type: ignore
            dtype = "float32"  # Default dtype
            if arg.torch_tensor is not None:
                dtype = str(arg.torch_tensor.dtype).split("torch.")[-1]

            loop_ir_tiling = ir.Tiling(
                name=arg.name,
                tensor_dims=tuple(tensor_dims),
                tile_dims=tuple(tile_dims),
                dim_names=tuple(dim_names),
                order=tuple(range(len(tensor_dims))),
                dtype=dtype,
                paged_metadata=arg.paged_metadata,
            )
            arg.tiling = loop_ir_tiling

        def create_transformed_tiling(
            self, tiling: TiledTensor, transform: TensorTransform
        ) -> TiledTensor:
            return TiledTensor(
                torch_tensor=tiling.torch_tensor,
                tiling=tiling.tiling,
                name=tiling.name,
                base_shape=tiling.base_shape,
                transforms=tiling.transforms + [transform],
                paged_metadata=tiling.paged_metadata,
            )

        def _parse_placeholder(self, node: torch.fx.Node):
            torch_arg = next(self.args_iter)
            name_to_use = node.name
            if self.input_names and self.placeholder_count < len(self.input_names):
                name_to_use = self.input_names[self.placeholder_count]

            if isinstance(torch_arg, torch.SymInt):
                torch_symint_name = torch_arg.node._graph_repr()
                symint_name = self.symint_names.get(
                    torch_symint_name, torch_symint_name
                )
                if torch_symint_name not in self.dim_manager.torch_symint_map:
                    ir_symint = ir.SymInt(symint_name)
                    self.dim_manager.torch_symint_map[torch_symint_name] = ir_symint
                else:
                    ir_symint = self.dim_manager.torch_symint_map[torch_symint_name]
                self.env[node.name] = ir_symint
                self.graph_inputs.append(ir_symint)
            elif isinstance(torch_arg, torch.Tensor):
                mapped_shape = [
                    self.dim_manager.from_torch_dim(d) for d in torch_arg.shape
                ]
                lazy_tiled_tensor = TiledTensor(
                    torch_arg, None, name_to_use, tuple(mapped_shape), []
                )
                self.env[node.name] = lazy_tiled_tensor
                self.graph_inputs.append(lazy_tiled_tensor)
            else:
                raise ValueError(f"Unknown argument type: {type(torch_arg)}")
            self.placeholder_count += 1

        def _parse_paged_load(self, node: torch.fx.Node):
            buffer, block_table = self._load_arg(node.args)
            S = block_table.current_shape
            N, *T = buffer.current_shape
            output_shape = (*S, *T)
            out_name = f"paged_{self.get_tmp_var(node.name)}"
            lazy_tiled_tensor = TiledTensor(
                torch.empty(0),
                None,
                out_name,
                tuple(output_shape),
                [],
                paged_metadata={"buffer": buffer, "block_table": block_table},
            )
            self.env[node.name] = lazy_tiled_tensor

        def _parse_call_function(self, node: torch.fx.Node):
            self.visitor.visit(node)

        def _parse_output(self, node: torch.fx.Node):
            # todo: deal with views on the output
            output = self._load_arg(node.args[0])
            if not isinstance(output, (list, tuple)):
                output = (output,)
            self.output_node = output
            self.env[node.name] = self.output_node

    parser = GraphParser(sample_inputs, input_names, symint_names)
    parser.parse_graph(gm)

    if DEBUG:
        print("\nbefore optimize")
        ir.pprint_node(parser.programs)
    programs = optimize.optimize(parser.programs)

    USED_SLIDING_MASK = any(
        node.target == torch.ops.loopfuse.sliding_mask.default
        for node in gm.graph.nodes
        if node.op == "call_function"
    )
    # todo: this is a temporary fix to avoid NaNs in fully masked rows when using sliding mask
    # need a stronger sparsity handling to get rid of this!
    if USED_SLIDING_MASK:
        programs = cleanup_passes.safe_exp(programs)

    if DEBUG:
        print("\nafter optimize")
        ir.pprint_node(programs)

    graph_inputs_fixed = [
        inp.tiling if isinstance(inp, TiledTensor) and inp.tiling is not None else inp
        for inp in parser.graph_inputs
    ]
    sorted_graph_tile_dims = sorted(parser.graph_tile_dims, key=lambda x: x.name)

    # Relabel output tensors to have clean names like "out_0", "out_1", etc.
    output_clean_names = {}
    if parser.output_node:
        num_outputs = len(parser.output_node)
        output_clean_names = {
            out.tiling: dataclasses.replace(
                out.tiling, name=f"out_{i}" if num_outputs > 1 else "out"
            )
            for i, out in enumerate(parser.output_node)
            if isinstance(out, TiledTensor) and out.tiling is not None
        }
    programs = cleanup_passes.relabel_outputs(programs, output_clean_names)

    created_tilings_fixed = [
        output_clean_names.get(t.tiling, t.tiling) if t is not None else None
        for t in parser.created_tilings
    ]
    loop_ir_output_fixed = tuple(
        output_clean_names.get(out.tiling, out.tiling)
        if isinstance(out, TiledTensor)
        else out
        for out in parser.output_node
    )

    kernel_programs = codegen_program(
        node=programs,
        graph_inputs=graph_inputs_fixed,
        graph_tile_dims=list(sorted_graph_tile_dims),
        created_tilings=created_tilings_fixed,
        loop_ir_output=loop_ir_output_fixed,
        new_dims_to_create=parser.dim_manager.new_dims_to_create,
        mode=mode,
    )
    assert len(kernel_programs) == 1, (
        f"Only one kernel program supported for now, found {len(kernel_programs)}"
    )
    kernel_program = kernel_programs[0]

    # todo: clean
    const_to_symint = parser.dim_manager.const_int_to_ir
    static_int_to_real_int = {v: k for k, v in const_to_symint.items()}
    realised_tile_dims = [
        static_int_to_real_int[sym_tile_dim]
        if sym_tile_dim.name.startswith("CONST_")
        else None
        for sym_tile_dim in sorted_graph_tile_dims
    ]

    # for debugging
    if USE_HARDCODE_KERNEL:
        with open("hardcode_kernel.py", "r") as f:
            code = f.read()
        kernel_program = KernelData(
            kernel_name="kernel_0",
            launch_name="launch_kernel_0",
            code=code,
        )

    if CODEGEN_ONLY:
        print("Codegen only mode, skipping triton kernel loading")
        print(kernel_program.code)

        def fn(*args, **kwargs):
            return kernel_program

        return fn

    def load_triton_kernel(kernel_data: KernelData):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            fname = f.name
            f.write(kernel_data.code)
        try:
            spec = importlib.util.spec_from_file_location("triton_kernel_module", fname)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules["triton_kernel_module"] = module
            spec.loader.exec_module(module)
            return getattr(module, kernel_data.launch_name)
        except Exception as e:
            raise e

    kernel_fn = load_triton_kernel(kernel_program)

    def optimized_fn(*args, **kwargs):
        result = kernel_fn(*args, realised_tile_dims)
        return result

    return optimized_fn
