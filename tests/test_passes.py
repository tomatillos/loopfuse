# testing load store elim

from loopfuse import ir
from loopfuse.compiler.passes.misc import eliminate_store_load


def test1():
    # store(X, i, full_tensor)
    # load(X, i)
    full_tensor = ir.Full((ir.SymInt("N"),), ir.Constant(1.0), "float32")
    X = ir.Tiling("X", (ir.SymInt("N"),), (ir.SymInt("tN"),), ("N",), "float32")
    i = ir.Variable("i")
    store_X = ir.Store(X, ir.Index((i,)), full_tensor)
    load_X = ir.Load(X, ir.Index((i,)))
    block = ir.Block([store_X, load_X])
    new_block = eliminate_store_load(block)
    assert len(new_block.body) == 1
    assert isinstance(new_block.body[0], ir.Full)
    assert new_block.body[0].shape == full_tensor.shape
    assert new_block.body[0].value == full_tensor.value
    assert new_block.body[0].dtype == full_tensor.dtype


def test2():
    # store(X, i, full_tensor)
    # load(X, i)
    # load(X, i)
    # should eliminate both
    full_tensor = ir.Full((ir.SymInt("N"),), ir.Constant(1.0), "float32")
    X = ir.Tiling("X", (ir.SymInt("N"),), (ir.SymInt("tN"),), ("N",), "float32")
    i = ir.Variable("i")
    store_X = ir.Store(X, ir.Index((i,)), full_tensor)
    load_X = ir.Load(X, ir.Index((i,)))
    load_X2 = ir.Load(X, ir.Index((i,)))
    block = ir.Block([store_X, load_X, load_X2])
    new_block = eliminate_store_load(block)
    ir.pprint_node(new_block)
    assert len(new_block.body) == 2


def test3():
    # store(X, i, full_tensor)
    # load(X, j)
    # load(X, i)
    # can't eliminate store, since it's used elsewhere
    full_tensor = ir.Full((ir.SymInt("N"),), ir.Constant(1.0), "float32")
    X = ir.Tiling("X", (ir.SymInt("N"),), (ir.SymInt("tN"),), ("N",), "float32")
    i = ir.Variable("i")
    j = ir.Variable("j")
    store_X = ir.Store(X, ir.Index((i,)), full_tensor)
    load_X_j = ir.Load(X, ir.Index((j,)))
    load_X_i = ir.Load(X, ir.Index((i,)))
    block = ir.Block([store_X, load_X_j, load_X_i])
    new_block = eliminate_store_load(block)
    assert len(block.body) == len(new_block.body)


def test4():
    # for i:
    #   for j:
    #     store(X, [i,j], full_tensor)
    #     load(X, [i,j])
    # for l:
    #   load(X, [l,1])
    # can't eliminate store, since it's used elsewhere
    full_tensor = ir.Full((ir.SymInt("N"), ir.SymInt("M")), ir.Constant(2.0), "float32")
    X = ir.Tiling(
        "X",
        (ir.SymInt("N"), ir.SymInt("M")),
        (ir.SymInt("tN"), ir.SymInt("tM")),
        ("N", "M"),
        "float32",
    )
    i = ir.Variable("i")
    j = ir.Variable("j")
    l = ir.Variable("l")
    store_X = ir.Store(X, ir.Index((i, j)), full_tensor)
    load_X_ij = ir.Load(X, ir.Index((i, j)))
    inner_loop = ir.Loop(
        j, ir.Constant(0), ir.SymInt("M"), ir.Block([store_X, load_X_ij])
    )
    outer_loop = ir.Loop(i, ir.Constant(0), ir.SymInt("N"), ir.Block([inner_loop]))
    load_X_l1 = ir.Load(X, ir.Index((l, ir.Constant(1))))
    l_loop = ir.Loop(l, ir.Constant(0), ir.SymInt("N"), ir.Block([load_X_l1]))
    block = ir.Block([outer_loop, l_loop])
    ir.pprint_node(block)
    new_block = eliminate_store_load(block)
    assert len(new_block.body[0].body[0].body) == len(block.body[0].body[0].body)
