from dataclasses import dataclass
from typing import Callable
import uuid

from loopfuse import ir
from loopfuse.compiler.helpers import find_loads, find_stores, substitute


def replace_load(block: ir.Block, replace_map: dict[uuid.UUID, ir.Node]) -> ir.Block:
    """Replaces loads."""
    visited: dict[uuid.UUID, ir.Node] = {}

    def _replace_load(node: ir.Node, stores: dict[uuid.UUID, ir.Node]) -> ir.Node:
        if node.id in visited:
            return visited[node.id]
        if node.id in stores:
            visited[node.id] = stores[node.id]
        else:
            visited[node.id] = node.map_children(lambda n: _replace_load(n, stores))
        return visited[node.id]

    new_block = _replace_load(block, replace_map)
    assert isinstance(new_block, ir.Block)
    return new_block


def hash_for_cse(node: ir.Node):
    """Hash function for CSE pass."""
    if isinstance(node, (ir.Loop, ir.Block, ir.Variable, ir.Zeros, ir.Full)):
        return (type(node).__name__, node.id)

    if hasattr(node, "__dataclass_fields__"):
        fields = []
        for field_name in node.__dataclass_fields__:
            if field_name == "id":
                continue
            value = getattr(node, field_name)
            if isinstance(value, ir.Node):
                fields.append(hash_for_cse(value))
            elif isinstance(value, list):
                fields.append(
                    tuple(
                        hash_for_cse(v) if isinstance(v, ir.Node) else v for v in value
                    )
                )
            elif isinstance(value, tuple):
                fields.append(
                    tuple(
                        hash_for_cse(v) if isinstance(v, ir.Node) else v for v in value
                    )
                )
            else:
                fields.append(value)
        return (type(node).__name__, *fields)

    raise NotImplementedError(f"Hash not implemented for {type(node)}")


def cse_pass(block: ir.Block) -> ir.Block:
    """Common subexpression elimination pass."""
    visited: dict[tuple, ir.Node] = {}

    def _walk(node: ir.Node) -> ir.Node:
        node_hash = hash_for_cse(node)
        if node_hash in visited:
            return visited[node_hash]
        new_node = node.map_children(lambda n: _walk(n))
        visited[node_hash] = new_node
        return new_node

    new_block = _walk(block)
    assert isinstance(new_block, ir.Block)
    return new_block


def remove_nested_blocks(block: ir.Block) -> ir.Block:
    """Removes nested blocks which sometimes arise."""
    new_body = []
    for stmt in block.body:
        if isinstance(stmt, ir.Block):
            # Recursively flatten nested blocks and extend the current block
            flattened = remove_nested_blocks(stmt)
            new_body.extend(flattened.body)
        else:
            new_body.append(stmt)
    return ir.Block(new_body)


def remove_store_from_block(block: ir.Block, store_tile, store_index) -> ir.Block:
    """Removes all occurrences of node from block (recursively)."""
    new_body: list[ir.Node] = []
    for stmt in block.body:
        if isinstance(stmt, ir.Store) and (stmt.tiling.name, stmt.index) == (
            store_tile,
            store_index,
        ):
            continue
        elif isinstance(stmt, ir.Loop):
            new_body.append(
                ir.Loop(
                    stmt.loop_var,
                    stmt.start,
                    stmt.end,
                    remove_store_from_block(stmt.body, store_tile, store_index),
                )
            )
        elif isinstance(stmt, ir.Block):
            new_body.append(remove_store_from_block(stmt, store_tile, store_index))
        else:
            new_body.append(stmt)
    return ir.Block(new_body)


def replace_load_from_data(
    block: ir.Block, load_tile: str, load_index: ir.Index, store_val: ir.Node
):
    """Replaces loads."""
    visited: dict[uuid.UUID, ir.Node] = {}

    def _replace_load(node: ir.Node, store_val: ir.Node) -> ir.Node:
        if node.id in visited:
            return visited[node.id]
        if isinstance(node, ir.Load) and (node.tiling.name, node.index) == (
            load_tile,
            load_index,
        ):
            # if load node has an other, we need to be careful and filter the store val
            if node.other:
                # store val becomes masked store val
                node_tiling = node.tiling
                tile_nontrivial_dims = 0
                in_leading = True
                mask_stmts = []
                for ix, tile_dim in enumerate(node_tiling.tile_dims):
                    if in_leading and tile_dim == ir.Constant(1):
                        continue
                    in_leading = False
                    tile_nontrivial_dims += 1
                    if tile_nontrivial_dims > 2:
                        raise ValueError("Only 2d tiles are supported")

                    # mask_stmt = f"{load_index[ix]} * {tile_dim} + tl.arange(0, {tile_dim}) < {node_tiling.tensor_dims[ix]}"
                    # in IR:
                    load_ix = node.index.index[ix]
                    offs_ix = ir.BinaryOp("*", load_ix, tile_dim)
                    arange_ix = ir.Arange(ir.Constant(0), tile_dim)
                    mask_lhs = ir.BinaryOp("+", offs_ix, arange_ix)
                    tensor_dim_ix = node_tiling.tensor_dims[ix]
                    mask_stmt = ir.BinaryOp("<", mask_lhs, tensor_dim_ix)
                    mask_stmts.append(mask_stmt)

                if len(mask_stmts) == 2:
                    # need to add the unsqueezes:
                    unsqueeze_offset = 0
                    if node_tiling.order[-1] < node_tiling.order[-2]:
                        unsqueeze_offset = 1
                    mask_stmt1 = ir.Unsqueeze(mask_stmts[0], 1 - unsqueeze_offset)
                    mask_stmt2 = ir.Unsqueeze(mask_stmts[1], 0 + unsqueeze_offset)
                    full_mask_stmt = ir.BinaryOp(
                        "&", mask_stmt1, mask_stmt2, name="bounds_mask"
                    )
                elif len(mask_stmts) == 1:
                    full_mask_stmt = mask_stmts[0]
                else:
                    raise ValueError(
                        f"Expected 1 or 2 mask statements, got {len(mask_stmts)}"
                    )

                store_val = ir.Where(full_mask_stmt, store_val, node.other)
                visited[node.id] = store_val
            else:
                visited[node.id] = store_val
                return store_val
        else:
            visited[node.id] = node.map_children(lambda n: _replace_load(n, store_val))
        return visited[node.id]

    new_block = _replace_load(block, store_val)
    assert isinstance(new_block, ir.Block)
    return new_block


def eliminate_store_load(block: ir.Block):
    changed = True
    while changed:
        stores = find_stores(block)
        loads = find_loads(block)
        for store_tile, store_index in stores:
            corresponding_loads = []
            can_eliminate = True
            for load_tile, load_index in loads:
                if load_tile == store_tile:
                    corresponding_loads.append((load_tile, load_index))
                    if load_index != store_index:
                        can_eliminate = False
                        break
            if not can_eliminate or not corresponding_loads:
                continue
            store_node = stores[(store_tile, store_index)]
            store_value = store_node.value
            block = replace_load_from_data(block, store_tile, store_index, store_value)
            block = remove_store_from_block(block, store_tile, store_index)
            changed = True
            break
        else:
            changed = False

    return cse_pass(block)


def remove_down_up_cast(block: ir.Block) -> ir.Block:
    """Eliminates redundant downcasting followed by upcasting operations.

    Case 1 - Direct cast chain elimination:
        x.to(bfloat16).to(float32) -> x

    Case 2 - Cast elimination through operations:
        c: float32
        y = x.to(float16) * c
        z = y.to(float32)
        ->
        z = x * c
    """
    dtype_order = {
        "float32": 3,
        "float16": 1,
        "bfloat16": 2,
    }
    visited: dict[uuid.UUID, ir.Node] = {}

    def rewrite_casts(node: ir.Node) -> ir.Node:
        if node.id in visited:
            return visited[node.id]
        if isinstance(node, ir.Cast):
            processed_operand = rewrite_casts(node.operand)
            downcast_found = [False]

            def _visit(n: ir.Node) -> ir.Node:
                if (
                    isinstance(n, ir.Cast)
                    and dtype_order[n.dtype] < dtype_order[node.dtype]
                ):
                    downcast_found[0] = True
                    return n.operand
                elif isinstance(n, ir.BinaryOp):
                    return ir.BinaryOp(n.op, _visit(n.lhs), _visit(n.rhs))
                elif isinstance(n, ir.Where) and isinstance(n.y, ir.Constant):
                    return ir.Where(n.condition, _visit(n.x), n.y)
                elif isinstance(n, ir.UnaryOp):
                    return ir.UnaryOp(n.op, _visit(n.operand))
                else:
                    return n

            walked_node = _visit(processed_operand)
            if downcast_found[0]:
                new_node = walked_node
            else:
                new_node = ir.Cast(processed_operand, node.dtype)

            visited[node.id] = new_node
            return new_node

        else:
            new_node = node.map_children(rewrite_casts)
            visited[node.id] = new_node
            return new_node

    new_block = rewrite_casts(block)
    new_block = cse_pass(new_block)
    assert isinstance(new_block, ir.Block)
    return new_block


def relabel_buffer_vars(block: ir.Block) -> ir.Block:
    """Walk the graph and count how many variables start with buffer_.
    Reduce the counter and relabel all buffer_ variables to buffer_0, buffer_1, etc."""
    buffers: list[ir.Variable] = []
    visited_find: set[uuid.UUID] = set()

    def _find_buffers(node: ir.Node):
        if node.id in visited_find:
            return
        visited_find.add(node.id)
        if isinstance(node, ir.Variable) and node.name.startswith("buffer_"):
            buffers.append(node)

        for child in node.children():
            _find_buffers(child)

    _find_buffers(block)

    buffer_names = sorted(
        list(set(b.name for b in buffers)), key=lambda name: int(name.split("_")[1])
    )
    buffer_map = {old_name: f"buffer_{i}" for i, old_name in enumerate(buffer_names)}

    visited_relabel: dict[uuid.UUID, ir.Node] = {}

    def _relabel(node: ir.Node) -> ir.Node:
        if node.id in visited_relabel:
            return visited_relabel[node.id]

        if isinstance(node, ir.Variable) and node.name in buffer_map:
            new_node = ir.Variable(name=buffer_map[node.name])
            visited_relabel[node.id] = new_node
            return new_node

        new_node = node.map_children(_relabel)
        visited_relabel[node.id] = new_node
        return new_node

    new_block = _relabel(block)
    assert isinstance(new_block, ir.Block)
    return new_block


def relabel_outputs(
    block: ir.Block, relabel_map: dict[ir.Tiling, ir.Tiling]
) -> ir.Block:
    """Relabel the outputs of the block."""
    visited: dict[uuid.UUID, ir.Node] = {}

    def _relabel(node: ir.Node) -> ir.Node:
        if node.id in visited:
            return visited[node.id]
        if isinstance(node, ir.Store) and node.tiling in relabel_map:
            new_tiling = relabel_map[node.tiling]
            new_index = _relabel(node.index)
            new_value = _relabel(node.value)
            new_node = ir.Store(new_tiling, new_index, new_value)
            visited[node.id] = new_node
            return new_node
        new_node = node.map_children(_relabel)
        visited[node.id] = new_node
        return new_node

    new_block = _relabel(block)
    return new_block


def eliminate_x_minus_x(block: ir.Block) -> ir.Block:
    """Eliminates x - x, checking for equality of lhs and rhs.
    Tracks variables last value for deeper elimination."""
    visited: dict[uuid.UUID, ir.Node] = {}
    variable_last_value: dict[ir.Variable, ir.Node] = {}

    def _walk(node: ir.Node) -> ir.Node:
        if node.id in visited:
            return visited[node.id]
        if isinstance(node, ir.BinaryOp) and node.op == "-":
            if node.lhs == node.rhs:
                new_node = ir.Constant(0)
                visited[node.id] = new_node
                return new_node
            elif variable_last_value.get(node.lhs, node.lhs) == variable_last_value.get(
                node.rhs, node.rhs
            ):
                new_node = ir.Constant(0)
                visited[node.id] = new_node
                return new_node
        if isinstance(node, ir.Assign):
            variable_last_value[node.target] = node.value
        new_node = node.map_children(_walk)
        visited[node.id] = new_node
        return new_node

    new_block = _walk(block)
    assert isinstance(new_block, ir.Block)
    return new_block


### Generic simplification rules


def binary_op_simplify(block: ir.Block, rules: list[tuple]) -> tuple[ir.Block, bool]:
    """Apply binary operation simplification rules. Returns (new_block, changed)."""
    visited: dict[uuid.UUID, ir.Node] = {}
    changed = False

    def _simplify(node: ir.Node) -> ir.Node:
        nonlocal changed
        if node.id in visited:
            return visited[node.id]

        if isinstance(node, ir.BinaryOp):
            for op, lhs_pattern, rhs_pattern, replacement in rules:
                if (
                    node.op == op
                    and _matches(node.lhs, lhs_pattern)
                    and _matches(node.rhs, rhs_pattern)
                ):
                    new_node = _apply_replacement(replacement, node.lhs, node.rhs)
                    visited[node.id] = new_node
                    changed = True
                    return new_node

        new_node = node.map_children(_simplify)
        visited[node.id] = new_node
        return new_node

    new_block = _simplify(block)
    assert isinstance(new_block, ir.Block)
    final_block = cse_pass(new_block)
    return final_block, changed


def _matches(node: ir.Node, pattern) -> bool:
    if pattern == "any":
        return True
    if isinstance(pattern, (int, float)):
        return isinstance(node, ir.Constant) and node.value == pattern
    if pattern == "inf":
        return isinstance(node, ir.Constant) and node.value == float("inf")
    elif pattern == "-inf":
        return isinstance(node, ir.Constant) and node.value == float("-inf")
    return False


def _apply_replacement(replacement, lhs: ir.Node, rhs: ir.Node) -> ir.Node:
    if replacement == "lhs":
        return lhs
    if replacement == "rhs":
        return rhs
    if isinstance(replacement, (int, float)):
        return ir.Constant(replacement)
    if replacement == "inf":
        return ir.Constant(float("inf"))
    elif replacement == "-inf":
        return ir.Constant(float("-inf"))
    return replacement


def apply_arithmetic_once(block: ir.Block) -> tuple[ir.Block, bool]:
    """Apply arithmetic simplification rules once. Returns (new_block, changed)."""
    rules = [
        # op, lhs_pattern, rhs_pattern, replacement
        ("*", "any", 0, 0),  # X * 0 -> 0
        ("*", 0, "any", 0),  # 0 * X -> 0
        ("*", "any", 1, "lhs"),  # X * 1 -> X
        ("*", 1, "any", "rhs"),  # 1 * X -> X
        ("*", "any", "inf", "inf"),  # X * inf -> inf
        ("*", "inf", "any", "inf"),  # inf * X -> inf
        ("+", "any", 0, "lhs"),  # X + 0 -> X
        ("+", 0, "any", "rhs"),  # 0 + X -> X
        ("*", "any", "-inf", "-inf"),  # X * -inf -> -inf
        ("*", "-inf", "any", "-inf"),  # -inf * X -> -inf
        ("//", "any", 1, "lhs"),  # X // 1 -> X
    ]

    return binary_op_simplify(block, rules)


@dataclass(frozen=True)
class SimplifyRule:
    condition: Callable[[ir.Node], bool]
    replacement: Callable[[ir.Node], ir.Node]


exp_zero = SimplifyRule(
    condition=lambda node: isinstance(node, ir.UnaryOp)
    and node.op in ("exp", "exp2")
    and node.operand == ir.Constant(0),
    replacement=lambda node: ir.Constant(1),
)

assign_self = SimplifyRule(
    condition=lambda node: isinstance(node, ir.Assign) and node.target == node.value,
    replacement=lambda node: node.value,
)

swap_div_to_mul_rule = SimplifyRule(
    condition=lambda node: isinstance(node, ir.BinaryOp) and node.op == "/",
    replacement=lambda node: ir.BinaryOp(
        "*", ir.UnaryOp("reciprocal", node.rhs), node.lhs
    ),
)

# Add these new rules after the existing ones
trivial_div = SimplifyRule(
    condition=lambda node: (
        isinstance(node, ir.BinaryOp)
        and node.op == "//"
        and isinstance(node.lhs, ir.SymInt)
        and isinstance(node.rhs, ir.SymInt)
        and node.lhs == node.rhs
    ),
    replacement=lambda node: ir.Constant(1),
)


double_where = SimplifyRule(
    condition=lambda node: (
        isinstance(node, ir.Where)
        and isinstance(node.x, ir.Where)
        and node.x.y == node.y
    ),
    replacement=lambda node: ir.Where(
        ir.BinaryOp("&", node.condition, node.x.condition), node.x.x, node.y
    ),
)


def _fold_where_constant_condition(node: ir.Node) -> bool:
    return (
        isinstance(node, ir.BinaryOp)
        and node.op == "*"
        and (
            (isinstance(node.lhs, ir.Where) and isinstance(node.rhs, ir.Constant))
            or (isinstance(node.rhs, ir.Where) and isinstance(node.lhs, ir.Constant))
        )
    )


def _fold_where_constant_replacement(node: ir.Node) -> ir.Node:
    assert isinstance(node, ir.BinaryOp)
    if isinstance(node.lhs, ir.Where) and isinstance(node.rhs, ir.Constant):
        return ir.Where(
            node.lhs.condition,
            ir.BinaryOp("*", node.lhs.x, node.rhs),
            ir.BinaryOp("*", node.lhs.y, node.rhs),
        )
    else:
        assert isinstance(node.rhs, ir.Where) and isinstance(node.lhs, ir.Constant)
        return ir.Where(
            node.rhs.condition,
            ir.BinaryOp("*", node.rhs.x, node.lhs),
            ir.BinaryOp("*", node.rhs.y, node.lhs),
        )


fold_where_constant = SimplifyRule(
    condition=_fold_where_constant_condition,
    replacement=_fold_where_constant_replacement,
)

flatten_trivial_loop = SimplifyRule(
    condition=lambda node: isinstance(node, ir.Loop)
    and node.start == ir.Constant(0)
    and node.end == ir.Constant(1),
    replacement=lambda node: ir.Block(
        [substitute(n, node.loop_var, node.start) for n in node.body.body]
    ),
)


cast_constant = SimplifyRule(
    condition=lambda node: isinstance(node, ir.Cast)
    and isinstance(node.operand, ir.Constant),
    replacement=lambda node: node.operand,
)


def _add_neg_one_mul_to_sub_condition(node: ir.Node) -> bool:
    """Checks for a + (-1 * b)."""
    if not (isinstance(node, ir.BinaryOp) and node.op == "+"):
        return False

    rhs = node.rhs
    if not (isinstance(rhs, ir.BinaryOp) and rhs.op == "*"):
        return False

    return isinstance(rhs.lhs, ir.Constant) and rhs.lhs.value == -1


add_neg_one_mul_to_sub = SimplifyRule(
    condition=_add_neg_one_mul_to_sub_condition,
    replacement=lambda node: ir.BinaryOp("-", node.lhs, node.rhs.rhs),
)


exp_where_to_where_exp = SimplifyRule(
    condition=lambda node: (
        isinstance(node, ir.Where)
        and isinstance(node.y, ir.Constant)
        and node.y.value == 0
        and isinstance(node.x, ir.UnaryOp)
        and node.x.op in ("exp", "exp2")
    ),
    replacement=lambda node: ir.UnaryOp(
        node.x.op,
        ir.Where(
            condition=node.condition,
            x=node.x.operand,
            y=ir.Constant(float("-inf")),
        ),
    ),
)


where_cast_commute = SimplifyRule(
    condition=lambda node: (
        isinstance(node, ir.Where)
        and isinstance(node.x, ir.Cast)
        and isinstance(node.y, ir.Constant)
    ),
    replacement=lambda node: ir.Cast(
        ir.Where(node.condition, node.x.operand, node.y), node.x.dtype
    ),
)


# todo: manage where statements in a more principled fashion
where_arith_inf = SimplifyRule(
    # where(mask, x-y, -inf) -> where(mask, x, -inf) - y
    # this is !not! 100% solid in general (e.g. with infinite values of x, y), but can ignore for now
    condition=lambda node: (
        isinstance(node, ir.Where)
        and isinstance(node.x, ir.BinaryOp)
        and node.x.op in ("-", "+")
        and node.y == ir.Constant(float("-inf"))
    ),
    replacement=lambda node: ir.BinaryOp(
        node.x.op,
        ir.Where(node.condition, node.x.lhs, ir.Constant(float("-inf"))),
        node.x.rhs,
    ),
)


def apply_rules_once(block: ir.Block) -> tuple[ir.Block, bool]:
    """Apply simplification rules once. Returns (new_block, changed)."""
    visited: dict[uuid.UUID, ir.Node] = {}
    changed = False

    rules = [
        exp_zero,
        assign_self,
        swap_div_to_mul_rule,
        trivial_div,
        double_where,
        fold_where_constant,
        flatten_trivial_loop,
        cast_constant,
        add_neg_one_mul_to_sub,
        exp_where_to_where_exp,
        where_cast_commute,
        where_arith_inf,
    ]

    def _simplify(node: ir.Node) -> ir.Node:
        nonlocal changed
        if node.id in visited:
            return visited[node.id]
        for rule in rules:
            if rule.condition(node):
                new_node = rule.replacement(node)
                visited[node.id] = new_node
                changed = True
                return new_node
        new_node = node.map_children(_simplify)
        visited[node.id] = new_node
        return new_node

    new_block = _simplify(block)
    assert isinstance(new_block, ir.Block)
    return new_block, changed


def simplify_combined(block: ir.Block) -> ir.Block:
    """Simplifies the block using both rule-based and arithmetic simplification."""
    current = block
    num_simplify_steps = 1000

    for _ in range(num_simplify_steps):
        overall_changed = False

        # Apply rule-based simplifications
        new_block, changed = apply_rules_once(current)
        if changed:
            overall_changed = True
            current = new_block

        # Apply arithmetic simplifications
        new_block, changed = apply_arithmetic_once(current)
        if changed:
            overall_changed = True
            current = new_block

        if not overall_changed:
            break

    return current


def safe_exp(block: ir.Block) -> ir.Block:
    """Have to be careful not to do x - (-inf) as leads to NaNs.
    This is temporary until we have more thorough sparsity analysis.
    Only needed for attention masks that block values at the start of a row (e.g. sliding mask).

    replaces exp2(x-y) with exp2(x-mask(y)) where mask(y) = where(y == -inf, 0, y)
    """
    visited: dict[uuid.UUID, ir.Node] = {}

    def _walk(node: ir.Node) -> ir.Node:
        if node.id in visited:
            return visited[node.id]

        new_node = node.map_children(_walk)

        if (
            isinstance(new_node, ir.UnaryOp)
            and new_node.op in ("exp", "exp2")
            and isinstance(new_node.operand, ir.BinaryOp)
            and new_node.operand.op == "-"
        ):
            sub_op = new_node.operand
            x = sub_op.lhs
            y = sub_op.rhs

            masked_y = ir.Where(
                ir.BinaryOp("==", y, ir.Constant(float("-inf"))),
                ir.Constant(0),
                y,
            )
            new_sub_op = ir.BinaryOp("-", x, masked_y)
            replacement_node = ir.UnaryOp(new_node.op, new_sub_op)

            visited[node.id] = replacement_node
            return replacement_node

        visited[node.id] = new_node
        return new_node

    final_block = _walk(block)
    assert isinstance(final_block, ir.Block)
    return cse_pass(final_block)
