import logging
from dataclasses import dataclass
from functools import reduce
import uuid

import z3  # type: ignore
import sympy as sp  # type: ignore

from loopfuse import ir
from loopfuse.compiler.sympy_converter import from_sympy


@dataclass(frozen=True)
class Interval:
    lo: z3.ArithRef  # symbolic lower bound
    hi: z3.ArithRef  # symbolic upper bound


op_to_fn = {
    "+": (lambda a, b: Interval(a.lo + b.lo, a.hi + b.hi)),
    "-": (lambda a, b: Interval(a.lo - b.hi, a.hi - b.lo)),
    "*": (lambda a, b: Interval(a.lo * b.lo, a.hi * b.hi)),  # assumes all positive
}


def interval_of(node, z3_vars):
    if isinstance(node, ir.Constant):
        v = z3.IntVal(node.value)
        return Interval(v, v)

    if isinstance(node, ir.Variable):
        v = z3_vars.setdefault(node.name, z3.Int(node.name))
        return Interval(v, v)  # degenerate interval [x, x]

    if isinstance(node, ir.SymInt):
        v = z3_vars.setdefault(node.name, z3.Int(node.name))
        return Interval(v, v)  # degenerate interval [x, x]

    if isinstance(node, ir.Arange):
        lo = interval_of(node.start, z3_vars).lo
        hi = interval_of(node.end, z3_vars).hi - 1
        return Interval(lo, hi)

    if isinstance(node, ir.BinaryOp) and node.op in {"+", "-", "*"}:
        a = interval_of(node.lhs, z3_vars)
        b = interval_of(node.rhs, z3_vars)
        return op_to_fn[node.op](a, b)

    # treat as no-op for now
    if isinstance(node, ir.Unsqueeze):
        a = interval_of(node.operand, z3_vars)
        return Interval(a.lo, a.hi)

    raise TypeError(f"unsupported node {type(node), node}")


def classify_predicate(cmp_node: ir.BinaryOp) -> tuple[z3.ExprRef, z3.ExprRef]:
    z3_vars: dict[str, z3.Int] = {}
    lhs_iv = interval_of(cmp_node.lhs, z3_vars)
    rhs_iv = interval_of(cmp_node.rhs, z3_vars)

    delta_lo = lhs_iv.lo - rhs_iv.hi
    delta_hi = lhs_iv.hi - rhs_iv.lo

    op = cmp_node.op
    if op == "<":
        cond_true = delta_hi < 0
        cond_false = delta_lo >= 0
    elif op == "<=":
        cond_true = delta_hi <= 0
        cond_false = delta_lo > 0
    elif op == ">":
        cond_true = delta_lo > 0
        cond_false = delta_hi <= 0
    elif op == ">=":
        cond_true = delta_lo >= 0
        cond_false = delta_hi < 0
    else:
        raise ValueError(f"unsupported comparison {op}")

    return z3.simplify(cond_true), z3.simplify(cond_false)


def z3_to_ir(expr):
    if z3.is_not(expr):
        arg = expr.arg(0)
        if z3.is_le(arg):
            return ir.BinaryOp(">", z3_to_ir(arg.arg(0)), z3_to_ir(arg.arg(1)))

    if z3.is_le(expr):
        return ir.BinaryOp("<=", z3_to_ir(expr.arg(0)), z3_to_ir(expr.arg(1)))
    if z3.is_gt(expr):
        return ir.BinaryOp(">", z3_to_ir(expr.arg(0)), z3_to_ir(expr.arg(1)))
    if z3.is_add(expr):
        args = [z3_to_ir(expr.arg(i)) for i in range(expr.num_args())]
        return reduce(lambda a, b: ir.BinaryOp("+", a, b), args)
    if z3.is_sub(expr):
        return ir.BinaryOp("-", z3_to_ir(expr.arg(0)), z3_to_ir(expr.arg(1)))
    if z3.is_mul(expr):
        return ir.BinaryOp("*", z3_to_ir(expr.arg(0)), z3_to_ir(expr.arg(1)))
    if z3.is_div(expr):
        return ir.BinaryOp("//", z3_to_ir(expr.arg(0)), z3_to_ir(expr.arg(1)))
    if z3.is_int_value(expr):
        return ir.Constant(expr.as_long())
    if z3.is_const(expr):
        return ir.Variable(str(expr))
    raise NotImplementedError(f"Cannot convert: {expr}")


@dataclass(frozen=True)
class CondVal:
    cond: z3.ExprRef
    value: ir.Node


def propagate_inequality(node: ir.BinaryOp) -> list[CondVal]:
    assert node.op in {"<", "<=", ">", ">="}
    true_cond, false_cond = classify_predicate(node)
    return [
        CondVal(true_cond, ir.Bool(True)),
        CondVal(false_cond, ir.Bool(False)),
    ]


def merge(primary_condval: CondVal, condval_list: list[CondVal]) -> list[CondVal]:
    combined_condvals = []
    for condval in condval_list:
        combined_condition = z3.simplify(z3.And(primary_condval.cond, condval.cond))
        combined_condvals.append(CondVal(combined_condition, condval.value))
    return combined_condvals


def cond_propagate(
    cond_val_list: list[CondVal], match_val: ir.Node, return_val: ir.Node
) -> list[CondVal]:
    """Propagates any conditions which match the match_val, returning the return_val"""
    return [
        CondVal(_cv.cond, return_val) for _cv in cond_val_list if _cv.value == match_val
    ]


def find_conditional_values(
    node, existing_conditions: dict[ir.Node, list[CondVal]]
) -> list[CondVal]:
    """returns a list of conditions which make the node constant."""

    # todo: rewrite with better pattern matching
    NEG_INF = ir.Constant(-float("inf"))
    ZERO = ir.Constant(0)
    ONE = ir.Constant(1)

    def _handle_binary_op(
        node: ir.BinaryOp, existing_conditions: dict[ir.Node, list[CondVal]]
    ) -> list[CondVal]:
        if node.op in {"<", "<=", ">", ">="}:
            return propagate_inequality(node)
        lhs_cond = visit(node.lhs)
        rhs_cond = visit(node.rhs)
        if node.op == "max":
            lhs_neg_inf = cond_propagate(lhs_cond, NEG_INF, node.rhs)
            rhs_neg_inf = cond_propagate(rhs_cond, NEG_INF, node.lhs)
            return lhs_neg_inf + rhs_neg_inf
        elif node.op == "*":
            lhs_zero = cond_propagate(lhs_cond, ZERO, ZERO)
            rhs_zero = cond_propagate(rhs_cond, ZERO, ZERO)
            lhs_one = cond_propagate(lhs_cond, ONE, node.rhs)
            rhs_one = cond_propagate(rhs_cond, ONE, node.lhs)
            lhs_neg_inf = cond_propagate(lhs_cond, NEG_INF, NEG_INF)
            rhs_neg_inf = cond_propagate(rhs_cond, NEG_INF, NEG_INF)
            return lhs_zero + rhs_zero + lhs_one + rhs_one + lhs_neg_inf + rhs_neg_inf
        elif node.op == "+":
            lhs_zero = [
                c
                for _cv in lhs_cond
                if _cv.value == ir.Constant(0)
                for c in merge(_cv, rhs_cond)
            ]
            rhs_zero = [
                c
                for _cv in rhs_cond
                if _cv.value == ir.Constant(0)
                for c in merge(_cv, lhs_cond)
            ]
            lhs_neg_inf = cond_propagate(lhs_cond, NEG_INF, NEG_INF)
            rhs_neg_inf = cond_propagate(rhs_cond, NEG_INF, NEG_INF)
            return lhs_zero + rhs_zero + lhs_neg_inf + rhs_neg_inf
        elif node.op == "-":
            lhs_eq_rhs = cond_propagate(lhs_cond, node.rhs, ZERO)
            lhs_neg_inf = cond_propagate(lhs_cond, NEG_INF, NEG_INF)
            rhs_eq_lhs = cond_propagate(rhs_cond, node.lhs, ZERO)
            extra_eqs = []
            if isinstance(node.lhs, ir.Variable) and node.lhs in existing_conditions:
                for _cond_val in rhs_cond:
                    for _existing_cond_val in existing_conditions[node.lhs]:
                        if _cond_val.value == _existing_cond_val.value:
                            simplifed_and_cond = z3.simplify(
                                z3.And(_cond_val.cond, _existing_cond_val.cond)
                            )
                            new_cond_val = CondVal(simplifed_and_cond, ZERO)
                            extra_eqs.append(new_cond_val)
            return lhs_eq_rhs + lhs_neg_inf + rhs_eq_lhs + extra_eqs
        elif node.op == "@":
            lhs_zero = cond_propagate(lhs_cond, ZERO, ZERO)
            rhs_zero = cond_propagate(rhs_cond, ZERO, ZERO)
            return lhs_zero + rhs_zero
        elif node.op == "&":
            lhs_false = cond_propagate(lhs_cond, ir.Bool(False), ir.Bool(False))
            rhs_false = cond_propagate(rhs_cond, ir.Bool(False), ir.Bool(False))
            lhs_true = cond_propagate(lhs_cond, ir.Bool(True), ir.Bool(True))
            rhs_true = cond_propagate(rhs_cond, ir.Bool(True), ir.Bool(True))
            return lhs_false + rhs_false + lhs_true + rhs_true
        return []

    visited: dict[uuid.UUID, list[CondVal]] = {}

    def visit(node: ir.Node) -> list[CondVal]:
        """Walks the graph and propagates conditions which makes the node constant."""
        if node.id in visited:
            return visited[node.id]

        result = []
        if isinstance(node, ir.BinaryOp):
            result = _handle_binary_op(node, existing_conditions)
        elif isinstance(node, ir.Where):
            mask_cond = visit(node.condition)
            mask_false = cond_propagate(mask_cond, ir.Bool(False), node.y)
            mask_true = cond_propagate(mask_cond, ir.Bool(True), node.x)
            # where(cond, x, x) = x
            x_cond = cond_propagate(visit(node.x), node.y, node.y)
            y_cond = cond_propagate(visit(node.y), node.x, node.x)
            result = mask_false + mask_true + x_cond + y_cond
        elif isinstance(node, ir.UnaryOp) and node.op in {"exp", "exp2"}:
            op_cond = visit(node.operand)
            exp_neg_inf = cond_propagate(op_cond, NEG_INF, ZERO)
            exp_zero = cond_propagate(op_cond, ZERO, ONE)
            result = exp_neg_inf + exp_zero
        elif isinstance(node, ir.Reduce):
            op_cond = visit(node.operand)
            if node.op == "max":
                max_neg_inf = cond_propagate(op_cond, NEG_INF, NEG_INF)
                result = max_neg_inf
            elif node.op == "sum":
                sum_zero = cond_propagate(op_cond, ZERO, ZERO)
                result = sum_zero
        elif isinstance(node, ir.Cast):
            result = visit(node.operand)
        elif isinstance(node, ir.Variable):
            result = [CondVal(True, node)]
        elif isinstance(node, ir.Constant):
            result = [CondVal(True, node)]

        visited[node.id] = result
        return result

    out = visit(node)
    return out


def find_conditions_for_assign_constant(
    node: ir.Assign, existing_conditions: dict[ir.Node, list[CondVal]]
) -> list[CondVal]:
    assert isinstance(node, ir.Assign)
    cond_vals = find_conditional_values(node.value, existing_conditions)
    return cond_vals


def try_eliminate_loop_iterations(block: ir.Block) -> ir.Block:
    def process_loop(inner_node: ir.Node) -> ir.Node:
        if isinstance(inner_node, ir.Block):
            return inner_node.map_children(process_loop)
        if isinstance(inner_node, ir.Loop):
            has_nested_loops = any(
                isinstance(stmt, ir.Loop) for stmt in inner_node.body.body
            )
            if has_nested_loops:
                return inner_node.map_children(process_loop)
            else:
                result = try_eliminate_loop_iterations_helper(inner_node)
                return result if result else inner_node
        return inner_node

    new_block = process_loop(block)
    assert isinstance(new_block, ir.Block)
    return new_block


def try_eliminate_loop_iterations_helper(loop: ir.Loop) -> ir.Loop | None:
    for body_node in loop.body:
        if not isinstance(body_node, ir.Assign):
            return None

    node_to_var: dict[ir.Node, list[ir.Node]] = {}
    for assign_stmt in loop.body:
        assert isinstance(assign_stmt, ir.Assign)
        node_to_var.setdefault(assign_stmt.value, []).append(assign_stmt.target)
    # partition all variables with the same last assign
    # variable -> list[variable] where they are equal at the end of each loop step
    assign_map: dict[ir.Node, list[CondVal]] = {}
    for _targets in node_to_var.values():
        for _target in _targets:
            assign_map[_target] = [CondVal(True, _t) for _t in _targets]

    unhandled_assigns = []
    for assign_stmt in loop.body:
        assert isinstance(assign_stmt, ir.Assign)
        result = find_conditions_for_assign_constant(assign_stmt, assign_map)
        if result:
            assign_map[assign_stmt.target] = result
        else:
            unhandled_assigns.append(assign_stmt)

    if unhandled_assigns:
        logging.debug(
            f"tried to eliminate loop iterations but found unhandled assigns: {len(unhandled_assigns)}"
        )
        return None

    fixed_variables = []
    var_to_fixing_conditions = {}  # var: list[conds] which fix the variable
    all_conditions = []
    for var in assign_map:
        fixing_conditions = []
        for cond_val in assign_map[var]:
            if cond_val.value == var or cond_val.value in fixed_variables:
                fixing_conditions.append(cond_val.cond)

        if not fixing_conditions:
            logging.debug(f"Couldn't fix variable {var}")
            return None
        fixed_variables.append(var)
        var_to_fixing_conditions[var] = fixing_conditions

        for condition in fixing_conditions:
            if condition not in all_conditions:
                all_conditions.append(condition)

    # find conditions which fix every variable -> these are the skip conditions
    z3_skip_conds = []
    for cond in all_conditions:
        if all(cond in fix_conds for fix_conds in var_to_fixing_conditions.values()):
            z3_skip_conds.append(z3.simplify(cond))

    lo = loop.start
    hi = loop.end
    found_skip = False
    for z3_skip_cond in z3_skip_conds:
        sympy_skip_cond = z3_to_sympy(z3_skip_cond)
        assert isinstance(loop.loop_var, ir.Variable)
        loop_var = sp.symbols(loop.loop_var.name, positive=True)
        res = sp.reduce_inequalities([sympy_skip_cond], loop_var)
        if not res.args:
            return None
        # check that the statement is of form loopvar < X
        if res.args[0] != loop_var:
            continue
        # todo extend to >= and <=
        if isinstance(res, sp.StrictGreaterThan):
            # loop_var > X => skip, so hi = min(hi, X)
            sympy_hi_skip = res.args[1]
            hi_skip = from_sympy(sympy_hi_skip, {}, {}, create_unknown_var=True)
            # todo: sort out the +1s do we need them in both cases. how does the rounding work in int cast
            hi_skip_plus_one = ir.BinaryOp("+", hi_skip, ir.Constant(1))
            hi_skip_plus_one_int = ir.Cast(hi_skip_plus_one, "int32")
            hi = ir.BinaryOp("min", hi, hi_skip_plus_one_int, name="range_end")
            found_skip = True
        elif isinstance(res, sp.StrictLessThan):
            # loop_var < X => skip, so lo = max(lo, X-1)
            sympy_lo_skip = res.args[1]
            lo_skip = from_sympy(sympy_lo_skip, {}, {}, create_unknown_var=True)
            lo_skip_minus_one = ir.BinaryOp("+", lo_skip, ir.Constant(1))
            lo_skip_minus_one_int = ir.Cast(lo_skip_minus_one, "int32")
            lo = ir.BinaryOp("max", lo, lo_skip_minus_one_int, name="range_start")
            found_skip = True
    if found_skip:
        new_loop = ir.Loop(loop.loop_var, lo, hi, loop.body)
        return new_loop

    return None


def z3_to_sympy(expr: z3.ExprRef, cache=None):
    """
    Recursively translate a Z3 AST into an equivalent SymPy expression.
    """
    if cache is None:
        cache = {}

    if expr in cache:
        return cache[expr]

    if z3.is_const(expr) and expr.num_args() == 0:
        if isinstance(expr, z3.IntNumRef):
            sym = expr.py_value()
        else:
            sym = sp.symbols(
                expr.decl().name(), positive=True
            )  # todo this might be too relaxed in general
        cache[expr] = sym
        return sym

    children = [z3_to_sympy(c, cache) for c in expr.children()]
    k = expr.decl().kind()

    op_map = {
        z3.Z3_OP_AND: sp.And,
        z3.Z3_OP_OR: sp.Or,
        z3.Z3_OP_NOT: sp.Not,
        z3.Z3_OP_IMPLIES: sp.Implies,
        z3.Z3_OP_XOR: sp.Xor,
        z3.Z3_OP_EQ: sp.Eq,
        z3.Z3_OP_LT: lambda a, b: a < b,
        z3.Z3_OP_LE: lambda a, b: a <= b,
        z3.Z3_OP_GT: lambda a, b: a > b,
        z3.Z3_OP_GE: lambda a, b: a >= b,
        z3.Z3_OP_ADD: lambda *args: sp.Add(*args),
        z3.Z3_OP_MUL: lambda *args: sp.Mul(*args),
        z3.Z3_OP_SUB: lambda a, *rest: sp.Add(a, *(-r for r in rest)),
        z3.Z3_OP_UMINUS: lambda a: -a,
        z3.Z3_OP_POWER: lambda a, b: a**b,
        z3.Z3_OP_DIV: lambda a, b: a / b,
        z3.Z3_OP_IDIV: lambda a, b: sp.floor(a / b),
        z3.Z3_OP_MOD: lambda a, b: sp.Mod(a, b),
    }

    if k in op_map:
        res = op_map[k](*children)
    elif k == z3.Z3_OP_DISTINCT:
        res = sp.And(
            *(
                sp.Ne(children[i], children[j])
                for i in range(len(children))
                for j in range(i + 1, len(children))
            )
        )
    else:
        raise NotImplementedError(f"Unsupported operator: {expr.decl()}")

    cache[expr] = res
    return res
