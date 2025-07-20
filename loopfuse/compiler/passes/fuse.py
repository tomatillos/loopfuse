from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
import logging
import uuid

from loopfuse import ir
from loopfuse.compiler.helpers import (
    find_loads,
    find_stores,
    find_edited_loop_vars,
    get_node_variables,
    substitute,
)
from loopfuse.compiler.passes.factor_acc import try_online_hoist_fuse


class FusionStatus(Enum):
    INIT = auto()
    NO_CHANGE = auto()
    FUSED = auto()
    FACTORED = auto()


def recursively_fuse(block: ir.Block) -> tuple[ir.Block, FusionStatus]:
    """Attempts to recursively fuse adjacent loops."""
    new_stmts: list[ir.Node] = []
    status = FusionStatus.NO_CHANGE
    for stmt in block.body:
        if status == FusionStatus.NO_CHANGE and isinstance(stmt, ir.Loop):
            body_block, body_status = recursively_fuse(stmt.body)
            if body_status != FusionStatus.NO_CHANGE:
                status = body_status
                new_stmts.append(
                    ir.Loop(stmt.loop_var, stmt.start, stmt.end, body_block)
                )
                continue
        new_stmts.append(stmt)

    if status == FusionStatus.NO_CHANGE:
        new_stmts, status = fuse_once_pass(new_stmts)
    return ir.Block(new_stmts), status


def fuse_once_pass(stmts: list[ir.Node]) -> tuple[list[ir.Node], FusionStatus]:
    """Attempts to fuse adjacent loops, first by standard loop fusion, if that fails try online-hoist rewrite.
    Returns as soon as there is a single successful fusion.
    Returns the new statements and whether any changes were made.
    """
    loop_positions = [i for i, s in enumerate(stmts) if isinstance(s, ir.Loop)]
    for i1, i2 in zip(loop_positions, loop_positions[1:]):
        loop1, loop2 = stmts[i1], stmts[i2]
        assert isinstance(loop1, ir.Loop) and isinstance(loop2, ir.Loop)

        if can_fuse(loop1, loop2):
            fused_loop = basic_loop_fuse(loop1, loop2)
            blocking = stmts[i1 + 1 : i2]
            blocking_reorder = can_move_blocking_statements(blocking, loop1, loop2)
            if blocking_reorder:
                reordered = (
                    stmts[:i1]
                    + blocking_reorder.stmts_before_fuse
                    + [fused_loop]
                    + blocking_reorder.stmts_after_fuse
                    + stmts[i2 + 1 :]
                )
                return reordered, FusionStatus.FUSED

        if loop2_transformed := try_online_hoist_fuse(loop1, loop2):
            hoisted_block = stmts[:i2] + [loop2_transformed] + stmts[i2 + 1 :]
            return hoisted_block, FusionStatus.FACTORED

    return stmts, FusionStatus.NO_CHANGE


def can_fuse(loop1: ir.Loop, loop2: ir.Loop) -> bool:
    """Decides whether two loops can be fused.

    Two loops can be fused if:
        1. They have the same range
        2. Any variable that is edited in loop1 is not used in loop2 and vice versa
        3. Any tensor load/stores must not interfere with later load/stores in the other loop.
        (3 can be made more general!)
    """
    if loop1.start != loop2.start or loop1.end != loop2.end:
        return False

    if _has_edit_conflict(loop1, loop2):
        return False

    if not _check_tensor_deps(loop1, loop2):
        return False

    return True


def basic_loop_fuse(loop1: ir.Loop, loop2: ir.Loop) -> ir.Loop:
    """Fuses two loops by combining their bodies, standardising the loop variable and removing duplicate names."""
    loop1_var_names = get_node_variable_names(loop1)
    loop1_vars = get_node_variables(loop1)

    loop2_deduped = substitute(loop2, loop2.loop_var, loop1.loop_var)
    loop2_relabelled = relabel_duplicate_vars(
        loop2_deduped, loop1_vars, loop1_var_names
    )

    assert isinstance(loop2_relabelled, ir.Loop)
    ret = loop1.body + loop2_relabelled.body
    return ir.Loop(
        loop_var=loop1.loop_var,
        start=loop1.start,
        end=loop1.end,
        body=ret,
    )


def _check_tensor_deps(loop1: ir.Loop, loop2: ir.Loop) -> bool:
    """Returns whether all load/store indices are compatible for fusion."""

    def _get_load_store_maps(loop: ir.Loop):
        loads = find_loads(loop)
        stores = find_stores(loop)
        load_map, store_map = defaultdict(list), defaultdict(list)
        for (tile, idx), _ in loads.items():
            load_map[tile].append(idx)
        for (tile, idx), _ in stores.items():
            store_map[tile].append(idx)
        return load_map, store_map

    loads1, stores1 = _get_load_store_maps(loop1)
    loads2, stores2 = _get_load_store_maps(loop2)

    def _incompatible(src_map, src_var, tgt_map, tgt_var) -> bool:
        for tile, src_indices in src_map.items():
            for idx1 in src_indices:
                for idx2 in tgt_map.get(tile, []):
                    if any(
                        (d1 == src_var) ^ (d2 == tgt_var)
                        for d1, d2 in zip(idx1.index, idx2.index)
                    ):
                        return True
        return False

    if (
        _incompatible(stores1, loop1.loop_var, loads2, loop2.loop_var)
        or _incompatible(stores1, loop1.loop_var, stores2, loop2.loop_var)
        or _incompatible(loads1, loop1.loop_var, stores2, loop2.loop_var)
    ):
        return False

    return True


def _has_edit_conflict(a: ir.Loop, b: ir.Loop) -> bool:
    loop1_vars = get_node_variables(a)
    loop2_vars = get_node_variables(b)
    loop1_edit_vars = find_edited_loop_vars(a)
    loop2_edit_vars = find_edited_loop_vars(b)
    return bool(loop1_edit_vars & loop2_vars) or bool(loop2_edit_vars & loop1_vars)


def get_node_variable_names(node: ir.Node) -> set[str]:
    return set([v.name for v in get_node_variables(node)])


def relabel_duplicate_vars(
    node: ir.Node,
    banned_loop_vars: set[ir.Variable],
    banned_loop_var_names: set[str],
) -> ir.Node:
    visited: dict[uuid.UUID, ir.Node] = {}

    def _relabel_duplicate_vars(node: ir.Node) -> ir.Node:
        if node.id in visited:
            return visited[node.id]
        if (
            isinstance(node, ir.Variable)
            and node.name in banned_loop_var_names
            and node not in banned_loop_vars
        ):
            name = node.name
            while name in banned_loop_var_names:
                if name[-1].isdigit():
                    name = name[:-1] + str(int(name[-1]) + 1)
                else:
                    name = name + "_1"
            # todo: what if this variable is used later outside of the loop
            # I suspect shouldn't rely on the strings
            # maybe we don't deduplicate here, but do in codegen?
            banned_loop_var_names.add(name)
            visited[node.id] = ir.Variable(name)
            return visited[node.id]
        visited[node.id] = node.map_children(lambda c: _relabel_duplicate_vars(c))
        return visited[node.id]

    return _relabel_duplicate_vars(node)


@dataclass
class BlockMoveResult:
    stmts_before_fuse: list[ir.Node]
    stmts_after_fuse: list[ir.Node]


def can_move_blocking_statements(
    statements: list[ir.Node], loop1: ir.Loop, loop2: ir.Loop
) -> BlockMoveResult | None:
    """Determines if all statements can be moved before loop1 or after loop2."""
    stmts_before_fuse: list[ir.Node] = []
    stmts_after_fuse: list[ir.Node] = []

    def get_load_tile_names(node: ir.Node) -> set[str]:
        return {_x[0] for _x in find_loads(node).keys()}

    def get_store_tile_names(node: ir.Node) -> set[str]:
        return {_x[0] for _x in find_stores(node).keys()}

    # for now, ban any loads/stores, can deal with that later if needed
    loop1_vars = get_node_variable_names(loop1)
    loop1_stores = get_store_tile_names(loop1)
    loop2_vars = get_node_variable_names(loop2)
    for stmt in statements:
        if stmt_loads := get_store_tile_names(stmt):
            logging.debug(f"Can't move blocking statements: stmt has stores: {stmt_loads}")
            return None
        stmt_loads = get_load_tile_names(stmt)
        stmt_vars = get_node_variable_names(stmt)
        if stmt_loads:
            # only does case where we move the stmt before loop1, stmt is a load, and doesn't conflict with loop1 stores
            if loop1_stores.isdisjoint(stmt_loads):
                stmts_before_fuse.append(stmt)
            else:
                logging.debug("Can't move blocking statements: stmt has loads")
                return None
        elif loop1_vars.isdisjoint(stmt_vars):
            stmts_before_fuse.append(stmt)
        elif loop2_vars.isdisjoint(stmt_vars):
            stmts_after_fuse.append(stmt)
        else:
            logging.debug("Can't move blocking statements: stmt has vars in both loops")
            return None
    return BlockMoveResult(stmts_before_fuse, stmts_after_fuse)
