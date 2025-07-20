import uuid

from loopfuse import ir


def find_loads(node: ir.Node) -> dict[tuple[str, ir.Index], ir.Load]:
    """Records (tiling name, index) -> load."""
    loads: dict[tuple[str, ir.Index], ir.Load] = {}

    def _walk(node: ir.Node):
        if isinstance(node, ir.Load):
            loads[(node.tiling.name, node.index)] = node
        for child in node.children():
            _walk(child)

    _walk(node)
    return loads


def find_stores(
    node: ir.Node, recurse_into_loops=True
) -> dict[tuple[str, ir.Index], ir.Store]:
    """Records (tiling name, index) -> store. Optionally does not recurse into loops."""
    stores: dict[tuple[str, ir.Index], ir.Store] = {}

    def _walk(node: ir.Node):
        if not recurse_into_loops and isinstance(node, ir.Loop):
            return
        if isinstance(node, ir.Store):
            stores[(node.tiling.name, node.index)] = node
        for child in node.children():
            _walk(child)

    _walk(node)
    return stores


def find_edited_loop_vars(loop: ir.Loop) -> set[ir.Variable]:
    """Finds all variables assigned to in a loop."""
    edited_vars: set[ir.Variable] = set()

    def _inner(node: ir.Node):
        if isinstance(node, ir.Assign):
            edited_vars.add(node.target)
        for child in node.children():
            _inner(child)

    _inner(loop)
    return edited_vars


# shape inference


def broadcast_shapes(
    shape1: tuple[ir.Node, ...], shape2: tuple[ir.Node, ...]
) -> tuple[ir.Node, ...]:
    """Broadcasts two shapes (torch/numpy style)."""

    def eq_1(x):
        return x == 1 or x == ir.Constant(1) or x == ir.SymInt("CONST_1")

    rev1 = shape1[::-1]
    rev2 = shape2[::-1]
    result: list[ir.Node] = []
    for i in range(max(len(rev1), len(rev2))):
        dim1 = rev1[i] if i < len(rev1) else ir.Constant(1)
        dim2 = rev2[i] if i < len(rev2) else ir.Constant(1)
        if eq_1(dim1):
            result.append(dim2)
        elif eq_1(dim2):
            result.append(dim1)
        elif dim1 == dim2:
            result.append(dim1)
        else:
            raise ValueError(f"Shapes {shape1} and {shape2} are not broadcastable")
    return tuple(result[::-1])


def matmul_shape(
    lhs: tuple[ir.Node, ...], rhs: tuple[ir.Node, ...]
) -> tuple[ir.Node, ...]:
    """Propagates the shape of a matmul."""
    if len(lhs) == 1 and len(rhs) == 1:
        assert lhs[0] == rhs[0]
        return ()
    elif len(lhs) == 2 and len(rhs) == 1:
        assert lhs[1] == rhs[0]
        return (lhs[0],)
    elif len(lhs) == 1 and len(rhs) == 2:
        assert lhs[0] == rhs[0]
        return (rhs[1],)
    else:
        assert lhs[-1] == rhs[-2], f"Shapes {lhs} and {rhs} are not matmulable"
        batch_shape = broadcast_shapes(lhs[:-2], rhs[:-2])
        return batch_shape + (lhs[-2], rhs[-1])


def infer_shape(node: ir.Node) -> tuple[ir.Node, ...]:
    """Infers the shape of a node.
    Traverses the graph to find statements which determine the shape, and propagates through operations.
    """
    if isinstance(node, ir.Load):
        # sort the blockshape by the order, and then return the last two
        tile_dims = node.tiling.tile_dims
        order = node.tiling.order
        assert len(order) == len(tile_dims)
        ordered_tile_dims = tuple(tile_dims[order.index(i)] for i in range(len(order)))
        block_shape = ordered_tile_dims[-2:]
        return block_shape
    elif isinstance(node, ir.BinaryOp):
        lhs_shape = infer_shape(node.lhs)
        rhs_shape = infer_shape(node.rhs)
        return (
            matmul_shape(lhs_shape, rhs_shape)
            if node.op == "@"
            else broadcast_shapes(lhs_shape, rhs_shape)
        )
    elif isinstance(node, ir.UnaryOp):
        return infer_shape(node.operand)
    elif isinstance(node, ir.Reduce):
        _shape = list(infer_shape(node.operand))
        axis = node.axis
        if axis < 0:
            axis += len(_shape)
        if 0 <= axis < len(_shape):
            _shape[axis] = ir.Constant(1)  # Currently reduces keep the reduction axis
        return tuple(_shape)
    elif isinstance(node, ir.Zeros):
        return node.shape
    elif isinstance(node, ir.Where):
        return infer_shape(node.x)
    elif isinstance(node, ir.Arange):
        return (node.end,)
    elif isinstance(node, ir.Unsqueeze):
        return (
            infer_shape(node.operand)[: node.axis]
            + (ir.Constant(1),)
            + infer_shape(node.operand)[node.axis :]
        )
    elif isinstance(node, ir.Variable):
        return (ir.Constant(1),)
    elif isinstance(node, ir.Cast):
        return infer_shape(node.operand)
    else:
        return ()


def get_node_variables(node: ir.Node) -> set[ir.Variable]:
    """Finds all variables in a node and it's children."""
    variables: set[ir.Variable] = set()

    def _get_variables(node: ir.Node):
        if isinstance(node, ir.Variable):
            variables.add(node)
        for child in node.children():
            _get_variables(child)

    _get_variables(node)
    return variables


def substitute(node: ir.Node, old_node: ir.Node, new_node: ir.Node) -> ir.Node:
    """Recursively substitutes old_node with new_node."""
    visited: dict[uuid.UUID, ir.Node] = {}

    def _inner(node: ir.Node) -> ir.Node:
        if node.id in visited:
            return visited[node.id]
        if old_node == node:
            visited[node.id] = new_node
            return new_node
        visited[node.id] = node.map_children(lambda n: _inner(n))
        return visited[node.id]

    return _inner(node)
