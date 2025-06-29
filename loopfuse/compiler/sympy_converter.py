import logging

import sympy  # type: ignore

from loopfuse import ir


def to_sympy(expr: ir.Node) -> tuple[sympy.Expr, dict[sympy.Symbol, ir.Load]]:
    sympy_var_to_load: dict[sympy.Symbol, ir.Load] = {}

    binary_ops = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "*": lambda x, y: x * y,
        "/": lambda x, y: x / y,
        "**": lambda x, y: x**y,
        "%": lambda x, y: sympy.Mod(x, y),
        "//": lambda x, y: sympy.floor(x / y),
    }

    unary_ops = {
        "exp": sympy.exp,
        "exp2": lambda x: 2**x,
        "reciprocal": lambda x: 1 / x,
    }

    def _to_sympy(node: ir.Node) -> sympy.Expr:
        if isinstance(node, ir.BinaryOp):
            op_func = binary_ops.get(node.op)
            if op_func is None:
                raise NotImplementedError(
                    f"Unsupported sympy binary operator: {node.op}"
                )
            return op_func(_to_sympy(node.lhs), _to_sympy(node.rhs))
        elif isinstance(node, ir.UnaryOp):
            op_func = unary_ops.get(node.op)
            if op_func is None:
                raise NotImplementedError(
                    f"Unsupported sympy unary operator: {node.op}"
                )
            return op_func(_to_sympy(node.operand))
        elif isinstance(node, ir.Variable):
            return sympy.Symbol(node.name)
        elif isinstance(node, ir.Constant):
            value = node.value
            if isinstance(value, int):
                return sympy.Integer(value)
            elif isinstance(value, float):
                return sympy.Float(value)
            else:
                raise NotImplementedError(f"Unsupported constant type: {type(value)}")
        elif isinstance(node, ir.Load):
            sympy_var = sympy.Symbol(node.tiling.name)
            sympy_var_to_load[sympy_var] = node
            return sympy_var
        elif isinstance(node, ir.Cast):
            return _to_sympy(node.operand)
        elif isinstance(node, ir.SymInt):
            return sympy.Symbol(node.name)
        else:
            raise NotImplementedError(f"Unknown node type: {type(node)}")

    return _to_sympy(expr), sympy_var_to_load


def from_sympy(
    expr: sympy.Expr,
    sympy_var_to_load: dict[sympy.Symbol, ir.Load],
    sympy_buf_to_ir_buf: dict[sympy.Symbol, ir.Variable],
    create_unknown_var: bool = False,
) -> ir.Node:
    def is_subtraction(args):
        # sympy represents (a - b) as (a + (-1)*b)
        return (
            len(args) == 2
            and isinstance(args[1], sympy.Mul)
            and len(args[1].args) == 2
            and args[1].args[0] == -1
        )

    def _from_sympy(expr: sympy.Expr) -> ir.Node:
        if isinstance(expr, sympy.Add):
            args = list(expr.args)
            if is_subtraction(args):
                return ir.BinaryOp(
                    "-", _from_sympy(args[0]), _from_sympy(args[1].args[1])
                )
            node = _from_sympy(args[0])
            for arg in args[1:]:
                node = ir.BinaryOp("+", node, _from_sympy(arg))
            return node
        elif isinstance(expr, sympy.Mul):
            args = list(expr.args)
            node = _from_sympy(args[0])
            for arg in args[1:]:
                node = ir.BinaryOp("*", node, _from_sympy(arg))
            return node
        elif isinstance(expr, sympy.Mod):
            return ir.BinaryOp(
                "%", _from_sympy(expr.args[0]), _from_sympy(expr.args[1])
            )
        elif isinstance(expr, sympy.floor):
            # Handle floor(a/b) -> a // b
            arg = expr.args[0]
            if isinstance(arg, sympy.Mul) and len(arg.args) == 2:
                # Check for a * (1/b) pattern which sympy uses for division
                if isinstance(arg.args[1], sympy.Pow) and arg.args[1].args[1] == -1:
                    return ir.BinaryOp(
                        "//", _from_sympy(arg.args[0]), _from_sympy(arg.args[1].args[0])
                    )
            elif hasattr(arg, "is_Mul") and arg.is_Mul:
                # Handle more complex multiplication cases
                numerator_terms = []
                denominator_terms = []
                for term in arg.args:
                    if isinstance(term, sympy.Pow) and term.args[1] == -1:
                        denominator_terms.append(term.args[0])
                    else:
                        numerator_terms.append(term)

                if denominator_terms and len(denominator_terms) == 1:
                    numerator = (
                        sympy.Mul(*numerator_terms)
                        if len(numerator_terms) > 1
                        else numerator_terms[0]
                    )
                    denominator = denominator_terms[0]
                    return ir.BinaryOp(
                        "//", _from_sympy(numerator), _from_sympy(denominator)
                    )
            # For other cases, convert to floor division by 1 (identity operation)
            return ir.BinaryOp("//", _from_sympy(arg), ir.Constant(1))
        elif isinstance(expr, sympy.exp):
            return ir.UnaryOp("exp", _from_sympy(expr.args[0]))
        elif isinstance(expr, sympy.Pow) and expr.args[0] == 2:
            return ir.UnaryOp("exp2", _from_sympy(expr.args[1]))
        elif isinstance(expr, sympy.Pow) and expr.args[1] == -1:
            return ir.UnaryOp("reciprocal", _from_sympy(expr.args[0]))
        elif isinstance(expr, sympy.Symbol):
            if expr in sympy_var_to_load:
                return sympy_var_to_load[expr]
            elif expr in sympy_buf_to_ir_buf:
                return sympy_buf_to_ir_buf[expr]
            elif create_unknown_var:
                logging.debug(f"Unknown symbol: {expr}, creating variable")
                return ir.Variable(expr.name)
            else:
                raise ValueError(f"Unknown symbol: {expr}")
        elif isinstance(expr, (sympy.Integer, sympy.Float)):
            return ir.Constant(
                expr.evalf() if isinstance(expr, sympy.Float) else int(expr)
            )
        else:
            raise ValueError(f"Unknown sympy type: {type(expr)}")

    return _from_sympy(expr)
