from loopfuse import ir
from loopfuse.compiler.passes.factor_acc import try_online_hoist_fuse
from loopfuse.compiler.passes.skip_loop_iterations import try_eliminate_loop_iterations
from loopfuse.compiler.passes.fuse import recursively_fuse
from loopfuse.torch_backend.ops import (
    build_reduce,
    build_binary_op,
    build_matmul,
    create_tiledtensor,
)
from loopfuse import optimize


def test_factor_accumulation():
    x_tiling = ir.Tiling("x", (ir.SymInt("N"),), (ir.Constant(32),), ("N",), "float32")

    loop_var = ir.Variable("i")
    x = ir.Load(x_tiling, ir.Index((loop_var,)))
    M = ir.Variable("M")
    acc = ir.Variable("acc")

    max_op = ir.BinaryOp("max", x, M)
    max_assign = ir.Assign(M, max_op)
    loop1 = ir.Loop(
        loop_var, ir.Constant(0), x_tiling.num_tiles[0], ir.Block([max_assign])
    )

    exp_op = ir.BinaryOp("*", ir.UnaryOp("exp", x), M)
    acc_plus = ir.BinaryOp("+", acc, exp_op)
    acc_assign = ir.Assign(acc, acc_plus)
    loop2 = ir.Loop(
        loop_var, ir.Constant(0), x_tiling.num_tiles[0], ir.Block([acc_assign])
    )

    # todo: fix this test
    result = try_online_hoist_fuse(loop1, loop2)
    assert result is not None
    assert isinstance(result.new_loop2, ir.Loop)
    assert len(result.post_loop_stmts) > 0

    # checking example that does not factor
    exp_op = ir.BinaryOp("+", ir.UnaryOp("exp", x), M)
    acc_plus = ir.BinaryOp("+", acc, exp_op)
    acc_assign = ir.Assign(acc, acc_plus)
    loop3 = ir.Loop(
        loop_var, ir.Constant(0), x_tiling.num_tiles[0], ir.Block([acc_assign])
    )
    result = try_online_hoist_fuse(loop1, loop3)
    assert result is None


def test_skip_loop_iterations():
    i = ir.Variable("i")
    tot = ir.Variable("tot")

    i_lt_5 = ir.BinaryOp("<=", i, ir.Constant(5))
    x = ir.Variable("x")
    where_op = ir.Where(i_lt_5, x, ir.Constant(0))

    tot_plus_where = ir.BinaryOp("+", tot, where_op)
    assign = ir.Assign(tot, tot_plus_where)

    loop = ir.Loop(i, ir.Constant(0), ir.Constant(10), ir.Block([assign]))

    block = ir.Block([loop])

    new_block = try_eliminate_loop_iterations(block)
    ir.pprint_node(new_block)

    assert isinstance(new_block, ir.Block)
    assert len(new_block.body) == 1
    assert isinstance(new_block.body[0], ir.Loop)
    assert isinstance(new_block.body[0].body, ir.Block)
    assert len(new_block.body[0].body.body) == 1
    assert isinstance(new_block.body[0].body.body[0], ir.If)

    # check the condition that it creates is correct
    generated_cond = new_block.body[0].body.body[0].condition
    assert generated_cond.op == "<="
    assert generated_cond.lhs.name == "i"
    assert generated_cond.rhs == ir.Constant(5)


def test_fuse_adjacent_loops():
    i = ir.Variable("i")
    j = ir.Variable("j")
    x = ir.Variable("x")
    y = ir.Variable("y")

    # Create two adjacent loops that can be fused
    # for i in range(10):
    #     x += i
    # for i in range(10):
    #     y += i
    loop1 = ir.Loop(
        i,
        ir.Constant(0),
        ir.Constant(10),
        ir.Block([ir.Assign(x, ir.BinaryOp("+", x, i))]),
    )

    loop2 = ir.Loop(
        i,
        ir.Constant(0),
        ir.Constant(10),
        ir.Block([ir.Assign(y, ir.BinaryOp("+", y, i))]),
    )

    # Create a nested loop that shouldn't be fused with the outer loops
    # for j in range(5):
    #     for i in range(10):
    #         x += i
    nested_loop = ir.Loop(
        j,
        ir.Constant(0),
        ir.Constant(5),
        ir.Block(
            [
                ir.Loop(
                    i,
                    ir.Constant(0),
                    ir.Constant(10),
                    ir.Block([ir.Assign(x, ir.BinaryOp("+", x, i))]),
                )
            ]
        ),
    )

    block = ir.Block([loop1, loop2, nested_loop])

    optimized, changed = recursively_fuse(block)
    ir.pprint_node(optimized)

    assert changed
    assert isinstance(optimized, ir.Block)
    assert len(optimized.body) == 2

    fused_loop = optimized.body[0]
    assert isinstance(fused_loop, ir.Loop)
    assert fused_loop.loop_var == i
    assert fused_loop.start == ir.Constant(0)
    assert fused_loop.end == ir.Constant(10)
    assert len(fused_loop.body.body) == 2

    nested_loop_opt = optimized.body[1]
    assert isinstance(nested_loop_opt, ir.Loop)
    assert nested_loop_opt.loop_var == j
    assert nested_loop_opt.start == ir.Constant(0)
    assert nested_loop_opt.end == ir.Constant(5)
    assert len(nested_loop_opt.body.body) == 1
    assert isinstance(nested_loop_opt.body.body[0], ir.Loop)


def count_loops(node: ir.Node) -> int:
    count = [0]

    def _walk(node: ir.Node):
        if isinstance(node, ir.Loop):
            count[0] += 1
        for child in node.children():
            _walk(child)

    _walk(node)
    return count[0]


def test_div_max_mul():
    # X/max(X) @ Y
    X_tiling = ir.Tiling(
        "X",
        (ir.SymInt("M"), ir.SymInt("N")),
        (ir.SymInt("tM"), ir.SymInt("tN")),
        ("M", "N"),
        "float32",
    )
    X = create_tiledtensor(X_tiling)
    Y_tiling = ir.Tiling(
        "Y",
        (ir.SymInt("N"), ir.SymInt("CONST_K")),
        (ir.SymInt("tN"), ir.SymInt("CONST_K")),
        ("N", "CONST_K"),
        "float32",
    )
    Y = create_tiledtensor(Y_tiling)

    max_loop = build_reduce(X, "max")
    div_loop = build_binary_op(X, max_loop.output, "/")
    matmul_loop = build_matmul("Z", div_loop.output, Y)
    full_program = ir.Block([max_loop.program, div_loop.program, matmul_loop.program])
    assert count_loops(full_program) == 7
    prg = optimize.optimize(full_program)
    # basic check the structure is right, fairly brittle!
    assert len(prg) == 1
    assert len(prg[0].body) == 5
    assert count_loops(prg) == 2
