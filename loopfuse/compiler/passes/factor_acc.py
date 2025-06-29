from dataclasses import dataclass
import logging
import uuid

import sympy  # type: ignore

from loopfuse import ir
from loopfuse.compiler import sympy_converter
from loopfuse.compiler.helpers import (
    find_edited_loop_vars,
    get_node_variables,
    infer_shape,
)


@dataclass
class FactoredAccumPattern:
    A: ir.Node
    B: ir.Node
    acc: ir.Variable


def try_online_hoist_fuse(loop1: ir.Loop, loop2: ir.Loop) -> ir.Block | None:
    """Attempts to apply the online-hoist fusion between loop1 and loop2.

    Returns loop2 transformed by the online-hoist fusion if possible, otherwise None.

    Specific case of algebraic manipulation to enable fusion.

    The setup:
    If loop2 references loop1's carried variables, we can't fuse them straightforwardly.
    But consider the case where loop2 is of the following form:
    ```
    for i:
        acc += Ai * Bi
    ```
    with Ai depending on the loop1 carried variables, but Bi not.
    Then we are able to rewrite the loop as:
    ```
    for i:
        acc *= A_{i-1}/A_i
        acc += Bi
    acc *= A_n
    ```
    In some cases, the ratio does not depend on the loop carried variables, enabling fusion!
    """
    if _check_loop_edited_vars_intersect(loop1, loop2):
        logging.debug("Can't online-hoist fuse: loop1 and loop2 edited vars intersect")
        return None

    # what's the high level logic?
    # 1. find the statements which depend on loop1's carried vars
    # 2. attempt to decompose the accumulate pattern into A*B form
    # 3. symbolically calculate + simplify the ratio, check it does not block
    # 4. fuse the loop with extra buffer variables
    carried_vars = _get_carried_var_deps(loop1, loop2)
    carried_var_dep_stmts = _get_carried_var_dep_stmts(loop2, carried_vars)
    if not carried_var_dep_stmts:
        logging.debug("shouldn't be here, no carried var dep stmts")
        return None

    pre_loop_stmts: list[ir.Node] = []
    post_loop_stmts: list[ir.Node] = []
    sympy_buf_to_ir_buf: dict[sympy.Symbol, ir.Variable] = {}
    for index, blocking_stmt in carried_var_dep_stmts:
        logging.debug(
            f"attempting to factor stmt {index} of {len(carried_var_dep_stmts)}"
        )
        accum_pattern = _parse_accum_pattern(blocking_stmt, carried_vars)
        if accum_pattern is None:
            logging.debug(
                "Can't online-hoist : loop2 assign stmt does not match the accum pattern"
            )
            return None

        # calculating ratio of A_{i-1}/A_i
        ratio_expr, sympy_buf_map, sympy_var_to_load = calc_increment_ratio(
            accum_pattern.A
        )
        buffer_inits, buffer_assigns, load_to_buf, sympy_buf_to_ir_buf = (
            _prepare_buffer_vars(sympy_buf_map, sympy_var_to_load, sympy_buf_to_ir_buf)
        )
        simplified_ratio = sympy_converter.from_sympy(
            ratio_expr, sympy_var_to_load, sympy_buf_to_ir_buf
        )
        if get_node_variables(simplified_ratio) & carried_vars:
            logging.debug(
                "Can't online-hoist: simplified ratio graph still depends on carried vars!"
            )
            return None

        # step 1: init the buffers
        pre_loop_stmts.extend(buffer_inits)

        # step 2: rewrite the acc pattern in the loop body
        loop2 = _rewrite_acc_pattern(
            accum_pattern,
            simplified_ratio,
            buffer_assigns,
            loop2,
            index,
        )

        # step 3: do the post loop step (acc *= A_n)
        post_loop_step = _build_post_loop(accum_pattern, load_to_buf, loop2)
        post_loop_stmts.extend(post_loop_step)

    transformed_loop2 = ir.Block(pre_loop_stmts + [loop2] + post_loop_stmts)
    return transformed_loop2


def _check_loop_edited_vars_intersect(
    loop1: ir.Loop, loop2: ir.Loop
) -> set[ir.Variable]:
    loop1_edited_vars = find_edited_loop_vars(loop1)
    loop2_edited_vars = find_edited_loop_vars(loop2)
    return loop1_edited_vars & loop2_edited_vars


def _get_carried_var_deps(loop1: ir.Loop, loop2: ir.Loop) -> set[ir.Variable]:
    loop1_edited_vars = find_edited_loop_vars(loop1)
    loop2_read_vars = get_node_variables(loop2)
    return loop1_edited_vars & loop2_read_vars


def _get_carried_var_dep_stmts(
    loop: ir.Loop, carried_vars: set[ir.Variable]
) -> list[tuple[int, ir.Node]]:
    """Returns a list of statements in loop which depend on the carried vars."""
    out = []
    for ix, stmt in enumerate(loop.body):
        # unwrap single statement blocks
        while isinstance(stmt, ir.Block) and len(stmt.body) == 1:
            stmt = stmt.body[0]
        if carried_vars & get_node_variables(stmt):
            out.append((ix, stmt))
    return out


def factor_out_carried_vars(
    node: ir.Node,
    carried_vars: set[ir.Variable],
) -> tuple[ir.Node | None, ir.Node | None]:
    """Rewrites nodes as A*B, where B does not depend on carried_vars.
    Returns (A, B).
    """
    node_vars = get_node_variables(node)
    if not node_vars & carried_vars:
        return (None, node)
    elif node_vars <= carried_vars:
        return (node, None)

    if isinstance(node, ir.BinaryOp) and node.op == "*":
        lhs_bad, lhs_good = factor_out_carried_vars(node.lhs, carried_vars)
        rhs_bad, rhs_good = factor_out_carried_vars(node.rhs, carried_vars)

        # helper function
        def maybe_mul(a: ir.Node | None, b: ir.Node | None) -> ir.Node:
            nodes = [x for x in [a, b] if x is not None]
            if len(nodes) == 2:
                return ir.BinaryOp("*", nodes[0], nodes[1])
            elif len(nodes) == 1:
                return nodes[0]
            raise Exception("shouldn't happen")

        bad_op: ir.Node = maybe_mul(lhs_bad, rhs_bad)
        good_op: ir.Node = maybe_mul(lhs_good, rhs_good)
        return (bad_op, good_op)
    elif isinstance(node, ir.BinaryOp) and node.op == "@":
        lhs_bad, lhs_good = factor_out_carried_vars(node.lhs, carried_vars)
        rhs_bad, rhs_good = factor_out_carried_vars(node.rhs, carried_vars)
        if not lhs_good or rhs_bad:
            return (node, None)
        assert rhs_good is not None
        # we can only factor this if lhs_bad acts row-wise on lhs_good
        if (
            lhs_bad
            and (lhs_bad_shape := infer_shape(lhs_bad))
            and lhs_bad_shape[-1] != 1
        ):
            return (node, None)
        return (lhs_bad, ir.BinaryOp("@", lhs_good, rhs_good))
    elif isinstance(node, ir.Reduce) and node.op == "sum":
        operand_bad, operand_good = factor_out_carried_vars(node.operand, carried_vars)
        if not operand_good:
            return (node, None)
        if not operand_bad:
            return (None, ir.Reduce(node.op, operand_good, node.axis))
        # For now, we'll assume operand_bad is scalar if it exists
        if infer_shape(operand_bad)[-1] != 1:
            return (node, None)
        return (operand_bad, ir.Reduce(node.op, operand_good, node.axis))
    elif isinstance(node, ir.Cast):
        # in general cast(a*b) != cast(a)*cast(b)
        # but we'll rock with it for now
        operand_bad, operand_good = factor_out_carried_vars(node.operand, carried_vars)
        if operand_bad is None:
            assert operand_good is not None
            return (None, ir.Cast(operand_good, node.dtype))
        elif operand_good is None:
            return (ir.Cast(operand_bad, node.dtype), None)
        return (ir.Cast(operand_bad, node.dtype), ir.Cast(operand_good, node.dtype))
    elif isinstance(node, ir.Where) and node.y in [
        ir.Constant(-float("inf")),
        ir.Constant(0),
    ]:
        # where(cond, x:=a*b, -inf|0) -> where(cond, a, -inf|0) * b
        cond_bad, cond_good = factor_out_carried_vars(node.x, carried_vars)
        return (cond_bad, ir.Where(node.condition, cond_good, node.y))
    else:
        return (node, None)


def _parse_accum_pattern(
    stmt: ir.Node, carried_vars: set[ir.Variable]
) -> FactoredAccumPattern | None:
    """Attempts to factor stmt of form
        acc = acc + x
    to
        acc = acc + a * b
    where b does not depend on carried_vars and a is scalar in the last row.
    If possible, returns FactoredAccumPattern(A, B, acc) else None.
    """
    if not isinstance(stmt, ir.Assign):
        logging.debug("Can't factor: loop2 assign stmt is not an assign")
        return None
    if not isinstance(stmt.value, ir.BinaryOp) or stmt.value.op != "+":
        return None
    A, B = factor_out_carried_vars(stmt.value.rhs, carried_vars)
    if not A:
        raise Exception("shouldn't happen!")
    if not B:
        logging.debug("Can't factor: loop2 assign stmt is not an accum pattern at B")
        return None
    return FactoredAccumPattern(A, B, stmt.target)


class BufferCounter:
    """Global buffer counter for uniquely naming buffers."""

    def __init__(self):
        self.counter = 0
        self.cache = {}

    def new_buffer_name(self, prefix="BUFFER"):
        name = f"{prefix}_{self.counter}"
        self.counter += 1
        return name

    def get_buffer_from_str(self, name: str) -> sympy.Symbol:
        if name in self.cache:
            return self.cache[name]
        new_name = self.new_buffer_name(prefix="buffer")
        self.cache[name] = sympy.Symbol(new_name)
        return self.cache[name]


buffer_counter = BufferCounter()


def calc_increment_ratio(
    A: ir.Node,
) -> tuple[ir.Node, dict[sympy.Symbol, sympy.Symbol], dict[sympy.Symbol, ir.Load]]:
    """For a node A, calculates the symbolic ratio of A_{i-1}/A_i.
    Returns the ratio and map from sympy variables to IR variables.
    """
    A_sympy, sympy_var_to_load = sympy_converter.to_sympy(A)
    # todo: don't sub all loads, only ones that change with the loop
    sympy_var_to_buffer_var = {
        v: buffer_counter.get_buffer_from_str(f"{v}_buffer") for v in sympy_var_to_load
    }
    A_delta_loopvar_sympy = A_sympy.subs(sympy_var_to_buffer_var)
    ratio = A_delta_loopvar_sympy / A_sympy
    simplified_ratio = sympy.simplify(ratio)
    return simplified_ratio, sympy_var_to_buffer_var, sympy_var_to_load


def _prepare_buffer_vars(
    sympy_buf_map: dict[sympy.Symbol, sympy.Symbol],
    sympy_var_to_load: dict[sympy.Symbol, ir.Load],
    sympy_buf_to_ir_buf: dict[sympy.Symbol, ir.Variable],
) -> tuple[
    list[ir.Node],
    list[ir.Node],
    dict[uuid.UUID, ir.Variable],
    dict[sympy.Symbol, ir.Variable],
]:
    buffer_inits: list[ir.Node] = []
    buffer_assigns: list[ir.Node] = []
    load_to_buf: dict[uuid.UUID, ir.Variable] = {}
    for sympy_var, sympy_var_buffer in sympy_buf_map.items():
        already_in = sympy_var_buffer in sympy_buf_to_ir_buf
        buffer_var = sympy_buf_to_ir_buf.get(
            sympy_var_buffer, ir.Variable(str(sympy_var_buffer))
        )
        load_node = sympy_var_to_load[sympy_var]
        _load_shape = infer_shape(load_node)
        _dtype = load_node.tiling.dtype
        load_to_buf[load_node.id] = buffer_var
        sympy_buf_to_ir_buf[sympy_var_buffer] = buffer_var
        # removes duplicate
        if not already_in:
            buffer_inits.append(
                ir.Assign(buffer_var, ir.Full(_load_shape, ir.Constant(1), _dtype))
            )
            buffer_assigns.append(ir.Assign(buffer_var, load_node))
    return buffer_inits, buffer_assigns, load_to_buf, sympy_buf_to_ir_buf


def _rewrite_acc_pattern(
    acc_pat: FactoredAccumPattern,
    simplified_ratio: ir.Node,
    buffer_assigns: list[ir.Node],
    loop2: ir.Loop,
    stmt_index: int,
) -> ir.Loop:
    acc_mul = ir.BinaryOp("*", acc_pat.acc, simplified_ratio)
    acc_plus = ir.BinaryOp("+", acc_mul, acc_pat.B)
    acc_assign = ir.Assign(acc_pat.acc, acc_plus)
    factored_stmts = [acc_assign]

    new_loop_2_body = (
        loop2.body.body[:stmt_index]
        + factored_stmts
        + loop2.body.body[stmt_index + 1 :]
        + buffer_assigns
    )
    return ir.Loop(loop2.loop_var, loop2.start, loop2.end, ir.Block(new_loop_2_body))


def eval_at_final_step(
    node: ir.Node,
    load_node_to_buffer_var: dict[uuid.UUID, ir.Variable],
    loop: ir.Loop,
) -> ir.Node:
    """Calculates the value of node at the final iteration of the loop, using the buffer variables"""
    visited: dict[uuid.UUID, ir.Node] = {}

    def _eval(node: ir.Node):
        if node.id in visited:
            return visited[node.id]
        if node.id in load_node_to_buffer_var:
            return load_node_to_buffer_var[node.id]
        if isinstance(node, ir.Variable) and node == loop.loop_var:
            return loop.end
        visited[node.id] = node.map_children(lambda c: _eval(c))
        return visited[node.id]

    return _eval(node)


def _build_post_loop(
    acc_pat: FactoredAccumPattern,
    load_to_buf: dict[uuid.UUID, ir.Variable],
    loop: ir.Loop,
) -> list[ir.Node]:
    A_n = eval_at_final_step(
        acc_pat.A,
        load_to_buf,
        loop,
    )
    return [ir.Assign(acc_pat.acc, ir.BinaryOp("*", acc_pat.acc, A_n))]
