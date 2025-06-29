from dataclasses import dataclass
import logging
import uuid
import textwrap
import collections

from loopfuse import ir
from loopfuse.compiler.helpers import infer_shape

# optional flag for gpus where triton doesn't support tf32
specify_tf32 = False
try:
    import torch

    if torch.cuda.is_available():
        specify_tf32 = torch.cuda.get_device_capability()[1] < 8
except ImportError:
    pass


BINARY_OP_TO_TRITON = {
    "+": "+",
    "-": "-",
    "*": "*",
    "/": "/",
    "//": "//",
    "%": "%",
    ">=": ">=",
    ">": ">",
    "<=": "<=",
    "<": "<",
    "&": "&",
    "==": "==",
}
FUNCTIONAL_BINARY_OPS = {
    "max": lambda a, b: f"tl.maximum({a}, {b})",
    "min": lambda a, b: f"tl.minimum({a}, {b})",
}
TF32_STRING = ", allow_tf32=USE_TF32" if specify_tf32 else ""
FUNCTIONAL_BINARY_OPS["@"] = lambda a, b: f"tl.dot({a}, {b}{TF32_STRING})"
UNARY_OP_TO_TRITON = {
    "exp": "tl.exp",
    "exp2": "tl.exp2",
    "log": "tl.log",
    "log2": "tl.log2",
    "reciprocal": "1/",
    "sin": "tl.sin",
    "cos": "tl.cos",
    "abs": "tl.abs",
}
REDUCE_OP_TO_TRITON = {
    "sum": lambda x, axis: f"tl.sum({x}, axis={axis}, keep_dims=True)",
    "max": lambda x, axis: f"tl.max({x}, axis={axis}, keep_dims=True)",
    "min": lambda x, axis: f"tl.min({x}, axis={axis}, keep_dims=True)",
}


@dataclass(frozen=True)
class LoadArgs:
    tiling: ir.Tiling | None  # might be untiled
    name: str
    ptr_name: str
    strides: list[str]


@dataclass
class KernelData:
    kernel_name: str
    launch_name: str
    code: str


@dataclass
class CodeGenContext:
    graph_inputs: list[ir.Tiling | ir.SymInt]
    graph_tile_dims: list[ir.SymInt]
    created_tilings: list[ir.Tiling]
    loop_ir_output: tuple[ir.Tiling, ...]
    new_dims_to_create: dict[ir.BinaryOp, ir.SymInt]
    symints_to_new_dims: dict[ir.SymInt, ir.BinaryOp]

    def resolve_symint_for_launch(self, node: ir.SymInt) -> str:
        if isinstance(node, float | int):
            return str(node)
        assert isinstance(node, ir.SymInt), f"Expected SymInt, got {node}"
        if node in self.graph_inputs:
            return node.name
        elif node in self.graph_tile_dims:
            return f"graph_tile_dims[{self.graph_tile_dims.index(node)}]"
        elif node.name.startswith("CONST_"):
            return str(int(node.name[len("CONST_") :]))
        elif node in self.symints_to_new_dims:
            return node_to_python_expr(
                self.symints_to_new_dims[node], self.resolve_symint_for_launch
            )
        else:
            raise ValueError(f"Cannot resolve SymInt {node.name} to a variable")


def codegen_program(
    node: ir.Node,
    graph_inputs: list[ir.Tiling | ir.SymInt],
    graph_tile_dims: list[ir.SymInt],
    created_tilings: list[ir.Tiling],
    loop_ir_output: tuple[ir.Tiling, ...],
    new_dims_to_create: dict[ir.BinaryOp, ir.SymInt],
    mode: str | None = None,
):
    assert isinstance(node, ir.Block)
    kernels = []
    symints_to_new_dims = {v: k for k, v in new_dims_to_create.items()}
    context = CodeGenContext(
        graph_inputs=graph_inputs,
        graph_tile_dims=graph_tile_dims,
        created_tilings=created_tilings,
        loop_ir_output=loop_ir_output,
        new_dims_to_create=new_dims_to_create,
        symints_to_new_dims=symints_to_new_dims,
    )
    for kernel_ix, stmt in enumerate(node.body):
        logging.debug(f"Codegen kernel {kernel_ix}")
        kernel_name = f"kernel_{kernel_ix}"
        assert isinstance(stmt, ir.Loop)
        generator = KernelGenerator(stmt, kernel_name, context, mode=mode)
        kernels.append(generator.generate())
    return kernels


class KernelGenerator:
    def __init__(
        self,
        loop_node: ir.Loop,
        kernel_name: str,
        context: CodeGenContext,
        mode: str | None = None,
    ):
        self.loop_node = loop_node
        self.kernel_name = kernel_name
        self.context = context
        self.launch_name = f"launch_{kernel_name}"
        self.mode = mode

        current = self.loop_node
        outer_loops = [current]
        while (
            isinstance(current.body, ir.Block)
            and len(current.body.body) == 1
            and isinstance(current.body.body[0], ir.Loop)
        ):
            current = current.body.body[0]
            outer_loops.append(current)

        self.outer_loops = outer_loops
        self.inner_loop_body = current.body
        self.loop_length_exprs = [loop.end for loop in outer_loops]
        self.load_args: list[LoadArgs] = []
        self.symint_tracker: dict[str, ir.SymInt] = {}

    def generate(self) -> KernelData:
        kernel_jit_src = self._generate_jit_function()
        kernel_launch_src = self._generate_launch_function()

        CODE_HEADER = "import torch\nimport triton\nimport triton.language as tl\nfrom triton.language.extra import libdevice\n"
        if specify_tf32:
            CODE_HEADER += "USE_TF32 = torch.cuda.get_device_capability()[1] >= 8\n"

        full_kernel_code = f"{CODE_HEADER}\n{kernel_jit_src}\n{kernel_launch_src}\n\n"
        return KernelData(self.kernel_name, self.launch_name, full_kernel_code)

    def _generate_jit_function(self) -> str:
        kernel_body, load_args, symint_tracker = codegen_kernel_body(
            self.inner_loop_body, indent=4
        )
        self.load_args = load_args
        self.symint_tracker = symint_tracker
        program_id_args = self._generate_program_id_logic()

        pointer_args = [
            f"{_la.ptr_name}, {', '.join(_la.strides)}" for _la in self.load_args
        ]
        constexpr_args = list(self.symint_tracker.keys())
        kernel_header = self._generate_kernel_header(
            pointer_args,
            constexpr_args,
        )
        return f"{kernel_header}\n{program_id_args}\n{kernel_body}"

    def _generate_grid_src(self) -> str:
        graph_input_symint_names = {
            node.name
            for node in self.context.graph_inputs
            if isinstance(node, ir.SymInt)
        }

        def resolve_symint_grid(node):
            if node.name in graph_input_symint_names:
                return node.name
            elif node.name.startswith("CONST_"):
                return str(int(node.name[len("CONST_") :]))
            elif node in self.context.graph_tile_dims:
                return f'meta["{node.name}"]'
            elif node in self.context.symints_to_new_dims:
                return node_to_python_expr(
                    self.context.symints_to_new_dims[node], resolve_symint_grid
                )
            else:
                raise ValueError(f"Cannot resolve SymInt {node.name} to a variable")

        result = [
            node_to_python_expr(expr, resolve_symint_grid, use_triton_cdiv=True)
            for expr in self.loop_length_exprs
        ]
        return " * ".join(result)

    def _generate_launch_function(self) -> str:
        # kernel launch src
        kernel_arg_defs = []
        tensors_to_create = []

        graph_input_names = [gi.name for gi in self.context.graph_inputs]

        for tiling_load_arg in self.load_args:
            name = tiling_load_arg.name
            tiling = tiling_load_arg.tiling
            if name in graph_input_names:
                ptr_variable = name
            elif tiling in self.context.created_tilings:
                ptr_variable = name
                new_tensor_dims = [
                    node_to_python_expr(_td, self.context.resolve_symint_for_launch)
                    for _td in tiling.tensor_dims
                ]
                tensors_to_create.append(
                    f'{ptr_variable} = torch.empty(({", ".join(new_tensor_dims)},), device="cuda", dtype=torch.{tiling.dtype})'
                )
            else:
                raise ValueError(f"Cannot find load arg {tiling}")

            ptr_name = tiling_load_arg.ptr_name
            kernel_arg_defs.append(f"{ptr_name}={ptr_variable}")
            for stride_ix, stride in enumerate(tiling_load_arg.strides):
                kernel_arg_defs.append(f"{stride}={ptr_variable}.stride({stride_ix})")

        for _symint in self.symint_tracker.values():
            if _symint in self.context.graph_inputs:
                kernel_arg_defs.append(f"{_symint.name}={_symint.name}")
            elif _symint in self.context.graph_tile_dims:
                if _symint.name.startswith("tile_"):
                    continue
                arg_pos = self.context.graph_tile_dims.index(_symint)
                kernel_arg_defs.append(f"{_symint.name}=graph_tile_dims[{arg_pos}]")
            elif _symint in self.context.symints_to_new_dims:
                expr = node_to_python_expr(
                    self.context.symints_to_new_dims[_symint],
                    self.context.resolve_symint_for_launch,
                )
                kernel_arg_defs.append(f"{_symint.name}={expr}")
            else:
                raise ValueError(f"Cannot find constexpr arg {_symint}")

        if specify_tf32:
            kernel_arg_defs.append("USE_TF32=USE_TF32")

        grid_src = self._generate_grid_src()

        return_src = (
            f"    return {', '.join(o.name for o in self.context.loop_ir_output)}"
        )

        indented_tensors_to_create = textwrap.indent(
            "\n".join(tensors_to_create), " " * 4
        )
        kernel_arg_str = ",\n".join(kernel_arg_defs)
        indented_kernel_arg_defs = textwrap.indent(kernel_arg_str, " " * 16)

        # deduplicate graph input names
        graph_input_names_dedup = []
        gi_names_count = {}
        for name in graph_input_names:
            if name in gi_names_count:
                gi_names_count[name] += 1
                graph_input_names_dedup.append(f"{name}_dup_{gi_names_count[name]}")
            else:
                gi_names_count[name] = 1
                graph_input_names_dedup.append(name)


        launch_args = graph_input_names_dedup + ["graph_tile_dims"]

        return textwrap.dedent(f"""
        def {self.launch_name}({", ".join(launch_args)}):
        {indented_tensors_to_create}
            grid = lambda meta: ({grid_src},)
            {self.kernel_name}[grid](
{indented_kernel_arg_defs}
            )
        {return_src}
        """)

    def _generate_program_id_logic(self):
        def to_triton_expr(expr: ir.Node):
            if isinstance(expr, ir.CDiv):
                return (
                    f"tl.cdiv({to_triton_expr(expr.lhs)}, {to_triton_expr(expr.rhs)})"
                )
            elif isinstance(expr, ir.SymInt):
                return expr.name
            elif isinstance(expr, ir.Constant):
                return str(expr.value)
            else:
                raise NotImplementedError(f"Cannot convert {expr} to triton expr")

        pid_args = ["# Program ID calculation"]
        n = len(self.outer_loops)
        if n == 1:
            assert hasattr(self.outer_loops[0].loop_var, "name")
            pid_args.append(f"{self.outer_loops[0].loop_var.name} = tl.program_id(0)")
        else:
            pid_args.append("pid = tl.program_id(0)")

            # reverse order is very basic heuristic to improve cache locality
            loops = self.outer_loops[::-1]
            loop_length_exprs_rev = [loop.end for loop in loops]

            loop_lengths = []
            for i, expr_node in enumerate(loop_length_exprs_rev):
                expr_str = to_triton_expr(expr_node)
                is_simple = isinstance(expr_node, (ir.SymInt, ir.Constant))
                if not is_simple:
                    var_name = f"grid_dim_{i}"
                    pid_args.append(f"{var_name} = {expr_str}")
                    loop_lengths.append(var_name)
                else:
                    loop_lengths.append(expr_str)

            pid_rem = "pid"
            for i in range(n - 1):
                loop_var = loops[i].loop_var
                assert hasattr(loop_var, "name")
                pid_args.append(f"{loop_var.name} = {pid_rem} % {loop_lengths[i]}")
                if i < n - 2:
                    new_rem = f"pid_rem_{i}"
                    pid_args.append(f"{new_rem} = {pid_rem} // {loop_lengths[i]}")
                    pid_rem = new_rem
                else:  # last iteration of the loop
                    pid_rem = f"{pid_rem} // {loop_lengths[i]}"

            # last one
            last_loop_var = loops[-1].loop_var
            assert hasattr(last_loop_var, "name")
            pid_args.append(f"{last_loop_var.name} = {pid_rem}")

        return "\n".join(4 * " " + line for line in pid_args) + "\n"

    def _generate_kernel_header(
        self,
        pointer_args: list[str],
        constexpr_args: list[str],
    ) -> str:
        tile_dims_to_autotune = [
            _arg for _arg in constexpr_args if _arg.startswith("tile_")
        ]

        tile_dim_config_outer_parts = []
        tile_dim_config_inner_parts = []
        for ix, tile_dim in enumerate(tile_dims_to_autotune):
            s_inner = f'"{tile_dim}": tile_var_{ix}'
            # todo: better hints for these
            s_outer = f"for tile_var_{ix} in [16, 32, 64, 128]"
            tile_dim_config_inner_parts.append(s_inner)
            tile_dim_config_outer_parts.append(s_outer)

        tile_dim_config_inner = ", ".join(tile_dim_config_inner_parts)
        indent = " " * 8
        tile_dim_config_outer = "".join(
            [f"\n{indent}{part}" for part in tile_dim_config_outer_parts]
        )

        if self.mode == "max-autotune":
            stages = [1, 2, 4]
            warps = [4, 8]
        else:
            stages = [1]
            warps = [4]

        autotune_stmt = f"""@triton.autotune(
    configs=[
        triton.Config({{{tile_dim_config_inner}}}, num_stages=s, num_warps=w)
        for s in {stages}
        for w in {warps}{tile_dim_config_outer}
    ],
    key=[],
)"""
        kernel_def_args = pointer_args + [
            f"{arg}: tl.constexpr" for arg in constexpr_args
        ]
        kernel_def_args.append("USE_TF32: tl.constexpr")
        args_str = "    " + ",\n    ".join(kernel_def_args)

        return f"""{autotune_stmt}
@triton.jit
def {self.kernel_name}(
{args_str}
):"""


def codegen_kernel_body(node: ir.Node, indent: int = 0):
    """Generates the code for the jit kernel (i.e. inside the @triton.jit function)"""
    node_visitor = IRNodeVisitor()
    node_visitor.visit(node)
    nodes = node_visitor.nodes
    usage_counts: collections.defaultdict[uuid.UUID, int] = collections.defaultdict(int)
    for _node in nodes:
        for child in _node.children():
            usage_counts[child.id] += 1

    visitor = TritonCodeGenerator(indent_level=indent, usage_counts=usage_counts)
    visitor.visit(node)
    code = "\n".join(visitor.code_lines)
    return code, visitor.load_args, visitor.symint_tracker


class IRVisitor:
    """
    Base class for visiting IR nodes.
    """

    def visit(self, node: ir.Node, *args, **kwargs):
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node, *args, **kwargs)

    def generic_visit(self, node: ir.Node, *args, **kwargs):
        # Fallback for nodes that don't have a visit method
        # We'll just visit their children
        if hasattr(node, "children"):
            for child in node.children():
                self.visit(child, *args, **kwargs)


class IRNodeVisitor(IRVisitor):
    def __init__(self):
        self.nodes = []
        self.visited = set()

    def visit(self, node: ir.Node):
        if node.id in self.visited:
            return
        self.visited.add(node.id)
        self.nodes.append(node)
        super().visit(node)


class TritonCodeGenerator(IRVisitor):
    def __init__(
        self, indent_level: int = 0, usage_counts: dict[uuid.UUID, int] | None = None
    ):
        self.code_lines: list[str] = []
        self.visited: dict[uuid.UUID, str] = {}
        self.load_args: list[LoadArgs] = []
        self.tensor_names_to_tilings: dict[str, ir.Tiling] = {}
        self.symint_tracker: dict[str, ir.SymInt] = {}
        self.used_load_names: dict[str, ir.Load] = {}
        self.indent_level = indent_level
        self.usage_counts = usage_counts if usage_counts is not None else {}
        self.name_counters: collections.defaultdict[str, int] = collections.defaultdict(
            int
        )
        self.op_to_name = {
            "+": "add",
            "-": "sub",
            "*": "mul",
            "/": "div",
            "//": "floordiv",
            "%": "mod",
            ">=": "ge",
            ">": "gt",
            "<=": "le",
            "<": "lt",
            "==": "eq",
            "&": "bitwise_and",
            "@": "dot",
            "max": "var_max",
            "min": "var_min",
        }
        self.descriptive_name_nodes = {
            ir.UnaryOp: lambda n: f"var_{n.op}",
            ir.Where: self._get_where_prefix,
            ir.Reduce: lambda n: f"reduce_{n.op}",
            ir.BinaryOp: self._get_binary_op_prefix,
            ir.Load: lambda n: f"{n.tiling.name.lower()}_block",
            ir.Arange: lambda n: "arange",
        }

    def _is_constant_expr(self, node: ir.Node) -> bool:
        if isinstance(node, ir.Constant):
            return True
        if isinstance(node, ir.UnaryOp):
            return self._is_constant_expr(node.operand)
        return False

    def _get_operand_name(self, node: ir.Node) -> str:
        if node.id in self.visited:
            name = self.visited[node.id]
            if name.endswith("_block"):
                return name.replace("_block", "")
            if name.startswith("dot_"):
                return name.replace("dot_", "")
            return name
        if isinstance(node, ir.BinaryOp) and node.op == "@":
            return self._get_binary_op_prefix(node)
        if isinstance(node, ir.Load):
            return node.tiling.name.lower()
        if isinstance(node, ir.Variable):
            return node.name
        return "op"

    def _get_where_prefix(self, node: ir.Where) -> str:
        x_name = self._get_operand_name(node.x)
        if x_name.endswith("_masked"):
            return x_name
        return f"{x_name}_masked"

    def _get_binary_op_prefix(self, node: ir.BinaryOp) -> str:
        if node.op == "@":
            lhs_name = self._get_operand_name(node.lhs)
            rhs_name = self._get_operand_name(node.rhs)
            return f"dot_{lhs_name}_{rhs_name}"
        if node.op == "*":
            # Traverse down the multiplication chain to find the most significant operand
            # This helps to generate names like `dot_q_k_scaled` instead of `op_scaled`
            # when we have expressions like `(dot(q, k) * c1) * c2`
            current_node: ir.Node = node
            while isinstance(current_node, ir.BinaryOp) and current_node.op == "*":
                lhs_is_const = isinstance(current_node.lhs, ir.Constant)
                rhs_is_const = isinstance(current_node.rhs, ir.Constant)
                if lhs_is_const and not rhs_is_const:
                    current_node = current_node.rhs
                elif not lhs_is_const:
                    current_node = current_node.lhs
                else:
                    break
            # if we have `tanh(dot_q_k * C1) * C2`, we want to get `dot_q_k`
            # and name it `dot_q_k_scaled`
            while isinstance(current_node, (ir.UnaryOp, ir.BinaryOp, ir.Cast)):
                if isinstance(current_node, ir.BinaryOp) and current_node.op == "@":
                    break

                if isinstance(current_node, ir.UnaryOp):
                    current_node = current_node.operand
                elif isinstance(current_node, ir.Cast):
                    current_node = current_node.operand
                elif isinstance(current_node, ir.BinaryOp) and current_node.op == "*":
                    lhs_is_const = self._is_constant_expr(current_node.lhs)
                    rhs_is_const = self._is_constant_expr(current_node.rhs)
                    if lhs_is_const and not rhs_is_const:
                        current_node = current_node.rhs
                    elif not lhs_is_const and not rhs_is_const:
                        if isinstance(current_node.rhs, ir.UnaryOp):
                            current_node = current_node.lhs
                        else:
                            current_node = current_node.rhs
                    elif not lhs_is_const:
                        current_node = current_node.lhs
                    else:
                        break
                else:
                    break

            op_name = self._get_operand_name(current_node)
            if op_name.endswith("_scaled"):
                return op_name
            return f"{op_name}_scaled"

        return self.op_to_name.get(node.op, "bin_op")

    def indented(self, s: str) -> str:
        return f"{self.indent_level * ' '}{s}"

    def get_tmp_var(self, node: ir.Node) -> str:
        if node.id in self.visited:
            raise Exception(f"Duplicate node: {node}")

        if hasattr(node, "name") and node.name:
            prefix = node.name
        else:
            prefix = "t"
            node_type = type(node)
            if node_type in self.descriptive_name_nodes:
                prefix = self.descriptive_name_nodes[node_type](node)

        var_idx = self.name_counters[prefix]
        self.name_counters[prefix] += 1
        if var_idx > 0:
            tmp_var = f"{prefix}_{var_idx}"
        else:
            tmp_var = prefix

        self.visited[node.id] = tmp_var
        return tmp_var

    def ret_helper(self, gen_tmp, node, op_stmt):
        if gen_tmp:
            tmp_var = self.get_tmp_var(node)
            self.code_lines.append(self.indented(f"{tmp_var} = {op_stmt}"))
            return tmp_var
        return op_stmt

    def visit(self, node: ir.Node, gen_tmp: bool = True, force_inline: bool = False):
        if force_inline:
            return super().visit(node, gen_tmp=False)

        if isinstance(node, (ir.Constant, ir.Variable, ir.SymInt)):
            return super().visit(node, gen_tmp=False)

        if node.id in self.visited:
            return self.visited[node.id]

        is_complex = (
            isinstance(node, (ir.Load, ir.Reduce, ir.Zeros, ir.Full, ir.Where))
            or (isinstance(node, ir.BinaryOp) and node.op in ["@", "max", "min"])
            or (hasattr(node, "name") and node.name)
        )

        used_multiple_times = self.usage_counts.get(node.id, 0) > 1

        # we create a var if the caller asks for one, or if the node is complex,
        # or if it's used multiple times. Otherwise, we inline.
        if gen_tmp or is_complex or used_multiple_times:
            return super().visit(node, gen_tmp=True)
        else:
            return super().visit(node, gen_tmp=False)

    def visit_Block(self, node: ir.Block, gen_tmp: bool = True):
        for body_node in node.body:
            self.visit(body_node, gen_tmp=True)
        return ""

    def visit_Zeros(self, node: ir.Zeros, gen_tmp: bool = True):
        shape = ", ".join(self.visit(dim, gen_tmp=False) for dim in node.shape)
        op_stmt = f"tl.zeros([{shape}], dtype=tl.{node.dtype})"
        return self.ret_helper(gen_tmp, node, op_stmt)

    def visit_Full(self, node: ir.Full, gen_tmp: bool = True):
        shape = ", ".join(self.visit(dim, gen_tmp=False) for dim in node.shape)
        value = self.visit(node.value, gen_tmp=False)
        op_stmt = f"tl.full([{shape}], {value}, dtype=tl.{node.dtype})"
        return self.ret_helper(gen_tmp, node, op_stmt)

    def visit_Constant(self, node: ir.Constant, gen_tmp: bool = True):
        if node.value == -float("inf"):
            return '-float("inf")'
        return str(node.value)

    def visit_Variable(self, node: ir.Variable, gen_tmp: bool = True):
        return node.name

    def visit_BinaryOp(self, node: ir.BinaryOp, gen_tmp: bool = True):
        lhs = self.visit(node.lhs, gen_tmp=False)
        rhs = self.visit(node.rhs, gen_tmp=False)
        if node.op == "@":
            if infer_shape(node.lhs)[0] == ir.SymInt("CONST_1"):
                op_stmt = f"tl.sum(tl.trans({lhs}) * {rhs}, 0)"
            else:
                op_stmt = FUNCTIONAL_BINARY_OPS[node.op](lhs, rhs)
            tmp_var = self.get_tmp_var(node)
            self.code_lines.append(self.indented(f"{tmp_var} = {op_stmt}"))
            return tmp_var
        elif node.op in BINARY_OP_TO_TRITON:
            op_stmt = f"{lhs} {BINARY_OP_TO_TRITON[node.op]} {rhs}"
            if not gen_tmp:
                op_stmt = f"({op_stmt})"
        elif node.op in FUNCTIONAL_BINARY_OPS:
            op_stmt = FUNCTIONAL_BINARY_OPS[node.op](lhs, rhs)
        else:
            raise NotImplementedError(f"Unsupported binary op: {node.op}")
        return self.ret_helper(gen_tmp, node, op_stmt)

    def visit_UnaryOp(self, node: ir.UnaryOp, gen_tmp: bool = True):
        operand_str = self.visit(node.operand, force_inline=True)
        if node.op == "square":
            op_stmt = f"({operand_str} * {operand_str})"
        elif node.op == "relu":
            op_stmt = f"tl.where({operand_str} > 0, {operand_str}, 0.0)"
        elif node.op == "tanh":
            op_stmt = f"(2 * tl.sigmoid(2 * {operand_str}) - 1)"
        else:
            op_str = UNARY_OP_TO_TRITON.get(node.op, f"{node.op}")
            op_stmt = f"{op_str}({operand_str})"
        return self.ret_helper(gen_tmp, node, op_stmt)

    def visit_Reduce(self, node: ir.Reduce, gen_tmp: bool = True):
        operand_str = self.visit(node.operand)
        tmp_var = self.get_tmp_var(node)
        op_stmt = REDUCE_OP_TO_TRITON[node.op](operand_str, node.axis)
        self.code_lines.append(self.indented(f"{tmp_var} = {op_stmt}"))
        return tmp_var

    def visit_Loop(self, node: ir.Loop, gen_tmp: bool = True):
        loop_var = self.visit(node.loop_var)
        start = self.visit(node.start, gen_tmp=False)
        end = self.visit(node.end, gen_tmp=False)
        self.code_lines.append("")
        self.code_lines.append(
            self.indented(f"for {loop_var} in range({start}, {end}):")
        )
        self.indent_level += 4
        for body_node in node.body:
            self.visit(body_node, gen_tmp=True)
        self.indent_level -= 4
        self.code_lines.append("")
        return ""

    def visit_If(self, node: ir.If, gen_tmp: bool = True):
        condition = self.visit(node.condition, gen_tmp=False)
        self.code_lines.append(self.indented(f"if {condition}:"))
        self.indent_level += 4
        for body_node in node.then_block:
            self.visit(body_node, gen_tmp=True)
        self.indent_level -= 4
        return ""

    def visit_Assign(self, node: ir.Assign, gen_tmp: bool = True):
        target = self.visit(node.target)
        force_inline = not target.startswith("buffer")  # fairly temporary hack...
        value = self.visit(node.value, force_inline=force_inline)
        self.visited[node.value.id] = target
        self.code_lines.append(self.indented(f"{target} = {value}"))
        return ""

    def visit_Load(self, node: ir.Load, gen_tmp: bool = True):
        if (
            node.tiling.name in self.tensor_names_to_tilings
            and node.tiling != self.tensor_names_to_tilings[node.tiling.name]
        ):
            raise Exception(f"Duplicate tiling: {node.tiling}")

        if node.tiling.paged_metadata:
            block_table = node.tiling.paged_metadata["block_table"]
            buffer = node.tiling.paged_metadata["buffer"]
            block_table_load_args = LoadArgs(
                tiling=None,
                name=block_table.name,
                ptr_name=f"{block_table.name}_ptr",
                strides=[
                    f"{block_table.name}_stride_{i}"
                    for i in range(len(block_table.base_shape))
                ],
            )
            buffer_load_args = LoadArgs(
                tiling=None,
                name=buffer.name,
                ptr_name=f"{buffer.name}_ptr",
                strides=[
                    f"{buffer.name}_stride_{i}" for i in range(len(buffer.base_shape))
                ],
            )
            self.load_args.append(block_table_load_args)
            self.load_args.append(buffer_load_args)

            # now we need to generate the "load" code from the index
            S = block_table.base_shape
            N, *T = buffer.base_shape
            order = node.tiling.order
            # todo: make this more robust
            assert order[: len(S) - 1] == tuple(range(len(S) - 1)), (
                f"Order {order} interferes with paging indexing"
            )
            index = node.index.index
            tensor_dims = node.tiling.tensor_dims
            tile_dims = node.tiling.tile_dims
            # block_table: int[*S]
            # buffer: float[N, *T]
            # returns: float[*S, *T]
            # where out[*s, *t] = buffer[block_table[*s], *t]
            block_table_index = tuple(
                self.visit(idx, gen_tmp=False) for idx in index[: len(S)]
            )
            block_table_bounding_dims = tuple(
                self.visit(dim, gen_tmp=False) for dim in tensor_dims[: len(S)]
            )
            block_table_tile_dims = tuple(
                self.visit(dim, gen_tmp=False) for dim in tile_dims[: len(S)]
            )
            block_ptr_stmt, block_mask_stmt = get_ptr_mask_stmt(
                block_table_bounding_dims,
                block_table_tile_dims,
                block_table_load_args.strides,
                block_table_index,
                block_table_load_args.ptr_name,
                order=None,
                paged_index=None,
                indent_level=self.indent_level,
            )
            ptr_prefix = f"{block_table.name.lower()}_ptrs"
            ptr_idx = self.name_counters[ptr_prefix]
            self.name_counters[ptr_prefix] += 1
            ptr_var = f"{ptr_prefix}" if ptr_idx == 0 else f"{ptr_prefix}_{ptr_idx}"
            self.code_lines.append(
                f"{' ' * self.indent_level}{ptr_var} = {block_ptr_stmt}"
            )

            if block_mask_stmt:
                mask_prefix = f"{block_table.name.lower()}_mask"
                mask_idx = self.name_counters[mask_prefix]
                self.name_counters[mask_prefix] += 1
                mask_var = (
                    f"{mask_prefix}" if mask_idx == 0 else f"{mask_prefix}_{mask_idx}"
                )
                self.code_lines.append(self.indented(f"{mask_var} = {block_mask_stmt}"))
                block_load_stmt = f"tl.load({ptr_var}, mask={mask_var})"
            else:
                block_load_stmt = f"tl.load({ptr_var})"

            buf_loc = f"{block_table_load_args.name}_loc"
            self.code_lines.append(self.indented(f"{buf_loc} = {block_load_stmt}"))

            # sort everything by order
            inverse_order = [order.index(i) for i in range(len(order))]
            tensor_dims = tuple(node.tiling.tensor_dims[i] for i in inverse_order)
            tile_dims = tuple(node.tiling.tile_dims[i] for i in inverse_order)
            index = tuple(node.index.index[i] for i in inverse_order)
            buffer_index = tuple(
                self.visit(idx, gen_tmp=False) for idx in index[len(S) - 1 :]
            )
            buffer_bounding_dims = tuple(
                self.visit(dim, gen_tmp=False) for dim in tensor_dims[len(S) - 1 :]
            )
            buffer_tile_dims = tuple(
                self.visit(dim, gen_tmp=False) for dim in tile_dims[len(S) - 1 :]
            )
            strides_ordered = [
                buffer_load_args.strides[i - len(S) + 1]
                for i in inverse_order[len(S) - 1 :]
            ]

            buffer_ptr_stmt, buffer_mask_stmt = get_ptr_mask_stmt(
                buffer_bounding_dims,
                buffer_tile_dims,
                strides_ordered,
                buffer_index,
                buffer_load_args.ptr_name,
                order=None,
                paged_index=PagedIndexData(buf_loc, order[len(S) - 1] - (len(S) - 1)),
                indent_level=self.indent_level,
            )

            ptr_prefix = f"{buffer.name.lower()}_ptrs"
            ptr_idx = self.name_counters[ptr_prefix]
            self.name_counters[ptr_prefix] += 1
            ptr_var = f"{ptr_prefix}" if ptr_idx == 0 else f"{ptr_prefix}_{ptr_idx}"
            self.code_lines.append(
                f"{' ' * self.indent_level}{ptr_var} = {buffer_ptr_stmt}"
            )
            if buffer_mask_stmt:
                other_stmt = f", other={node.other.value}" if node.other else ""
                mask_prefix = f"{buffer.name.lower()}_mask"
                mask_idx = self.name_counters[mask_prefix]
                self.name_counters[mask_prefix] += 1
                mask_var = (
                    f"{mask_prefix}" if mask_idx == 0 else f"{mask_prefix}_{mask_idx}"
                )
                self.code_lines.append(
                    self.indented(f"{mask_var} = {buffer_mask_stmt}")
                )
                op_stmt = f"tl.load({ptr_var}, mask={mask_var}{other_stmt})"
            else:
                op_stmt = f"tl.load({ptr_var})"

            tmp_var = self.get_tmp_var(node)
            self.code_lines.append(self.indented(f"{tmp_var} = {op_stmt}"))
            return tmp_var
        else:
            index = tuple(self.visit(idx, gen_tmp=False) for idx in node.index.index)
            tile_load_args = codegen_load_args(node.tiling)
            self.load_args.append(tile_load_args)
            tensor_dims = tuple(
                self.visit(dim, gen_tmp=False) for dim in node.tiling.tensor_dims
            )
            tile_dims = tuple(
                self.visit(dim, gen_tmp=False) for dim in node.tiling.tile_dims
            )
            ptr_stmt, mask_stmt = get_ptr_mask_stmt(
                tensor_dims,
                tile_dims,
                tile_load_args.strides,
                index,
                tile_load_args.ptr_name,
                node.tiling.order,
                indent_level=self.indent_level,
            )
            ptr_prefix = f"{node.tiling.name.lower()}_ptrs"
            ptr_idx = self.name_counters[ptr_prefix]
            self.name_counters[ptr_prefix] += 1
            ptr_var = f"{ptr_prefix}" if ptr_idx == 0 else f"{ptr_prefix}_{ptr_idx}"
            self.code_lines.append(f"{' ' * self.indent_level}{ptr_var} = {ptr_stmt}")

            if mask_stmt:
                other_stmt = f", other={node.other.value}" if node.other else ""
                mask_prefix = f"{node.tiling.name.lower()}_mask"
                mask_idx = self.name_counters[mask_prefix]
                self.name_counters[mask_prefix] += 1
                mask_var = (
                    f"{mask_prefix}" if mask_idx == 0 else f"{mask_prefix}_{mask_idx}"
                )
                self.code_lines.append(self.indented(f"{mask_var} = {mask_stmt}"))
                op_stmt = f"tl.load({ptr_var}, mask={mask_var}{other_stmt})"
            else:
                op_stmt = f"tl.load({ptr_var})"
            tmp_var = self.get_tmp_var(node)
            self.code_lines.append(self.indented(f"{tmp_var} = {op_stmt}"))
            self.code_lines.append("")
            return tmp_var

    def visit_Store(self, node: ir.Store, gen_tmp: bool = True):
        value = self.visit(node.value, gen_tmp=False)
        index = [self.visit(idx, gen_tmp=False) for idx in node.index.index]
        if (
            node.tiling.name in self.tensor_names_to_tilings
            and node.tiling != self.tensor_names_to_tilings[node.tiling.name]
        ):
            raise Exception(f"Duplicate tiling: {node.tiling}")
        tile_load_args = codegen_load_args(node.tiling)
        self.load_args.append(tile_load_args)
        tensor_dims = tuple(
            self.visit(dim, gen_tmp=False) for dim in node.tiling.tensor_dims
        )
        tile_dims = tuple(
            self.visit(dim, gen_tmp=False) for dim in node.tiling.tile_dims
        )
        ptr_stmt, mask_stmt = get_ptr_mask_stmt(
            tensor_dims,
            tile_dims,
            tile_load_args.strides,
            index,
            tile_load_args.ptr_name,
            None,
            indent_level=self.indent_level,
        )

        ptr_prefix = f"{node.tiling.name.lower()}_ptrs"
        ptr_idx = self.name_counters[ptr_prefix]
        self.name_counters[ptr_prefix] += 1
        ptr_var = f"{ptr_prefix}" if ptr_idx == 0 else f"{ptr_prefix}_{ptr_idx}"
        self.code_lines.append(f"{' ' * self.indent_level}{ptr_var} = {ptr_stmt}")

        if mask_stmt:
            mask_prefix = f"{node.tiling.name.lower()}_mask"
            mask_idx = self.name_counters[mask_prefix]
            self.name_counters[mask_prefix] += 1
            mask_var = (
                f"{mask_prefix}" if mask_idx == 0 else f"{mask_prefix}_{mask_idx}"
            )
            self.code_lines.append(self.indented(f"{mask_var} = {mask_stmt}"))
            stmt = f"tl.store({ptr_var}, {value}, mask={mask_var})"
        else:
            stmt = f"tl.store({ptr_var}, {value})"
        self.code_lines.append(self.indented(stmt))
        return ""

    def visit_Index(self, node: ir.Index, gen_tmp: bool = True):
        return ", ".join(self.visit(idx, gen_tmp=False) for idx in node.index)

    def visit_SymInt(self, node: ir.SymInt, gen_tmp: bool = True):
        self.symint_tracker[node.name] = node
        return node.name

    def visit_Where(self, node: ir.Where, gen_tmp: bool = True):
        condition = self.visit(node.condition, gen_tmp=False)
        x = self.visit(node.x, gen_tmp=True)
        y = self.visit(node.y, gen_tmp=True)
        op_stmt = f"tl.where({condition}, {x}, {y})"
        return self.ret_helper(gen_tmp, node, op_stmt)

    def visit_Arange(self, node: ir.Arange, gen_tmp: bool = True):
        start = self.visit(node.start, gen_tmp=False)
        end = self.visit(node.end, gen_tmp=False)
        op_stmt = f"tl.arange({start}, {end})"
        return self.ret_helper(gen_tmp, node, op_stmt)

    def visit_Unsqueeze(self, node: ir.Unsqueeze, gen_tmp: bool = True):
        operand = self.visit(node.operand, gen_tmp=False)
        ndims = len(infer_shape(node.operand))
        dims = [":" for _ in range(ndims + 1)]
        dims[node.axis] = "None"
        op_stmt = f"{operand}[{', '.join(dims)}]"
        return self.ret_helper(gen_tmp, node, op_stmt)

    def visit_Cast(self, node: ir.Cast, gen_tmp: bool = True):
        operand = self.visit(node.operand, gen_tmp=False)
        op_stmt = f"{operand}.to(tl.{node.dtype})"
        return self.ret_helper(gen_tmp, node, op_stmt)

    def visit_CDiv(self, node: ir.CDiv, gen_tmp: bool = True):
        lhs = self.visit(node.lhs, gen_tmp=False)
        rhs = self.visit(node.rhs, gen_tmp=False)
        op_stmt = f"tl.cdiv({lhs}, {rhs})"
        return self.ret_helper(gen_tmp, node, op_stmt)


def codegen_load_args(tiling: ir.Tiling) -> LoadArgs:
    return LoadArgs(
        tiling=tiling,
        name=tiling.name,
        ptr_name=f"{tiling.name}_ptr",
        strides=[f"{tiling.name}_stride_{i}" for i in range(len(tiling.tensor_dims))],
    )


@dataclass
class PagedIndexData:
    offset: str
    pos: int


def get_ptr_mask_stmt(
    bounding_dims,
    tile_dims,
    stride_names,
    index,
    ptr_name,
    order=None,
    paged_index=None,
    indent_level=0,
):
    if order is None:
        order = tuple(range(len(bounding_dims)))

    tile_ndim = 0
    in_leading = True
    for tile_dim in tile_dims:
        if in_leading and tile_dim == "1":
            continue
        in_leading = False
        tile_ndim += 1
    assert tile_ndim <= 2, f"Maximum 2d tiles are supported, have {tile_ndim}"

    ptr_stmts = [ptr_name]
    mask_stmts = []
    cur_tile_size = 0

    ndims = len(bounding_dims)

    def get_full_offset_stmt(ix, td):
        if ix == "0":
            offset = f"tl.arange(0, {td})"
        else:
            offset = f"({ix} * {td} + tl.arange(0, {td}))"
        return offset

    for oi in range(ndims):
        # reverse the permutation
        i = order.index(oi)
        offset = index[i]
        in_paged_index = False
        if paged_index and i == paged_index.pos:
            in_paged_index = True
            offset = paged_index.offset

        if tile_dims[i] != "1" and not in_paged_index:
            offset = get_full_offset_stmt(index[i], tile_dims[i])

        stride = stride_names[i]
        bounding_dim = bounding_dims[i]

        ptr_stmt = f"{stride} * {offset}"

        if tile_dims[i] != "1" and tile_ndim == 2:
            if cur_tile_size == 0:
                a, b = ":", "None"
            else:
                a, b = "None", ":"
            ptr_stmt = f"({ptr_stmt})[{a}, {b}]"
            if in_paged_index:
                offset = get_full_offset_stmt(index[i], tile_dims[i])
            mask_stmt = f"({offset} < {bounding_dim})[{a}, {b}]"
            if index[i] != "0":
                mask_stmts.append(mask_stmt)
            cur_tile_size += 1
        elif tile_dims[i] != "1" and tile_ndim == 1:
            mask_stmt = f"({offset} < {bounding_dim})"
            mask_stmts.append(mask_stmt)
            cur_tile_size += 1

        ptr_stmts.append(ptr_stmt)

    if len(ptr_stmts) > 1:
        first, *rest = ptr_stmts
        indentation = " " * (indent_level + 4)
        bracket_indentation = " " * (indent_level)
        rest_str = "".join([f"\n{indentation}+ {term}" for term in rest])
        ptr_stmt = f"({first}{rest_str}\n{bracket_indentation})"
    else:
        ptr_stmt = ptr_stmts[0]

    if len(mask_stmts) > 1:
        first, *rest = mask_stmts
        indentation = " " * (indent_level + 4)
        bracket_indentation = " " * (indent_level)
        rest_str = "".join([f"\n{indentation}& {term}" for term in rest])
        mask_stmt = f"({first}{rest_str}\n{bracket_indentation})"
    elif mask_stmts:
        mask_stmt = mask_stmts[0]
    else:
        mask_stmt = ""
    return ptr_stmt, mask_stmt


def node_to_python_expr(node, resolve_symint_fn, use_triton_cdiv=False):
    if isinstance(node, ir.SymInt):
        return resolve_symint_fn(node)
    elif isinstance(node, ir.BinaryOp):
        # Special handling for pretty printing
        if node.op == "+":
            if (
                isinstance(node.rhs, ir.BinaryOp)
                and node.rhs.op == "*"
                and isinstance(node.rhs.lhs, ir.Constant)
                and node.rhs.lhs.value == -1
            ):
                # a + (-1 * b) -> a - b
                lhs = node_to_python_expr(node.lhs, resolve_symint_fn, use_triton_cdiv)
                rhs_of_mul = node_to_python_expr(
                    node.rhs.rhs, resolve_symint_fn, use_triton_cdiv
                )
                return f"({lhs} - {rhs_of_mul})"
            if isinstance(node.rhs, ir.Constant) and node.rhs.value < 0:
                lhs = node_to_python_expr(node.lhs, resolve_symint_fn, use_triton_cdiv)
                return f"({lhs} - {-node.rhs.value})"

        if node.op == "*":
            if isinstance(node.lhs, ir.Constant) and node.lhs.value == -1:
                rhs = node_to_python_expr(node.rhs, resolve_symint_fn, use_triton_cdiv)
                return f"-({rhs})"
            if isinstance(node.rhs, ir.Constant) and node.rhs.value == -1:
                lhs = node_to_python_expr(node.lhs, resolve_symint_fn, use_triton_cdiv)
                return f"-({lhs})"

        lhs = node_to_python_expr(node.lhs, resolve_symint_fn, use_triton_cdiv)
        rhs = node_to_python_expr(node.rhs, resolve_symint_fn, use_triton_cdiv)
        if node.op in ("//", "*", "+", "-", "%", "=="):
            return f"({lhs} {node.op} {rhs})"
        else:
            raise NotImplementedError(f"Unknown binary op: {node.op}")
    elif isinstance(node, int):
        return str(node)
    elif isinstance(node, ir.Constant):
        return str(node.value)
    elif isinstance(node, ir.CDiv):
        lhs = node_to_python_expr(node.lhs, resolve_symint_fn)
        rhs = node_to_python_expr(node.rhs, resolve_symint_fn)
        if use_triton_cdiv:
            return f"triton.cdiv({lhs}, {rhs})"
        else:
            return f"({lhs} + {rhs} - 1) // {rhs}"
    raise NotImplementedError(f"Cannot generate code for node: {node}")
