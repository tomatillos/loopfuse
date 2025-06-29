import logging

from loopfuse import ir
from loopfuse.compiler.passes.fuse import FusionStatus, recursively_fuse
from loopfuse.compiler.passes import cleanup_passes
from loopfuse.compiler.passes.skip_loop_iterations import try_eliminate_loop_iterations


def pre_fusion_passes(program: ir.Block) -> ir.Block:
    """Initial passes that happen once at the start of optimisation."""
    program = cleanup_passes.simplify_combined(program)
    program = cleanup_passes.cse_pass(program)
    return program


def post_fusion_passes(program: ir.Block) -> ir.Block:
    """Cleanup passes that run after fusion is complete."""
    cleanup_pass_functions = [
        cleanup_passes.remove_down_up_cast,
        cleanup_passes.eliminate_x_minus_x,
        cleanup_passes.simplify_combined,
        try_eliminate_loop_iterations,
        cleanup_passes.simplify_combined,
        cleanup_passes.relabel_buffer_vars,
        cleanup_passes.cse_pass,
    ]

    for pass_fn in cleanup_pass_functions:
        program = pass_fn(program)

    return program


def optimize(program: ir.Block) -> ir.Block:
    """Full optimization pass."""
    program = pre_fusion_passes(program)

    counter = 0
    MAX_STEPS = 100
    while counter < MAX_STEPS:
        counter += 1
        program_before_passes = str(program)
        logging.debug(f"optimize pass {counter}")

        # Fuse as much as possible
        fusion_status = FusionStatus.INIT
        while fusion_status != FusionStatus.NO_CHANGE:
            program, fusion_status = recursively_fuse(program)

        # Apply other passes
        program = cleanup_passes.eliminate_store_load(program)
        program = cleanup_passes.simplify_combined(program)

        if str(program) == program_before_passes:
            break

    program = post_fusion_passes(program)

    return program
