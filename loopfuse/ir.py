from __future__ import annotations
from dataclasses import dataclass, field, fields
import uuid
from typing import Callable, Iterator


@dataclass(frozen=True)
class Node:
    id: uuid.UUID = field(default_factory=uuid.uuid4, init=False, repr=False)
    name: str | None = field(default=None, repr=False, kw_only=True)

    def children(self) -> list[Node]:
        # return any node fields
        return [
            getattr(self, _field.name)
            for _field in fields(self)
            if isinstance(getattr(self, _field.name), Node)
        ]

    def map_children(self, f: Callable[[Node], Node]):  # -> Self:
        new_attrs = {}
        existing_attrs = {
            _field.name: getattr(self, _field.name)
            for _field in fields(self)
            if _field.name not in ["id"]
        }
        for _field in fields(self):
            field_entry = getattr(self, _field.name)
            if isinstance(field_entry, Node):
                new_attrs[_field.name] = f(field_entry)
        if new_attrs:
            existing_attrs.update(new_attrs)
            return self.__class__(**existing_attrs)
        return self

    def __str__(self) -> str:
        return pprint_node(self, print_stdout=False)


@dataclass(frozen=True)
class Block(Node):
    body: list[Node]

    def __iter__(self) -> Iterator[Node]:
        return iter(self.body)

    def __getitem__(self, ix: int) -> Node:
        return self.body[ix]

    def __len__(self) -> int:
        return len(self.body)

    def __add__(self, other: Block) -> Block:
        def flatten_block(block: Block) -> list[Node]:
            flattened = []
            for stmt in block.body:
                if isinstance(stmt, Block):
                    flattened.extend(stmt.body)
                else:
                    flattened.append(stmt)
            return flattened

        new_block_body = flatten_block(self) + flatten_block(other)
        return Block(new_block_body)

    def children(self) -> list[Node]:
        return self.body

    def map_children(self, f) -> Block:
        # flattens as it maps
        out = []
        for child in self.body:
            fchild = f(child)
            if isinstance(fchild, Block):
                out.extend(fchild.body)
            else:
                out.append(fchild)
        return Block(out)


@dataclass(frozen=True)
class Variable(Node):
    name: str

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass(frozen=True)
class Constant(Node):
    value: int | float

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other) -> bool:
        if isinstance(other, Constant):
            return self.value == other.value
        if isinstance(other, int):
            return self.value == other
        return False


@dataclass(frozen=True)
class Zeros(Node):
    shape: tuple[Node, ...]
    dtype: str


@dataclass(frozen=True)
class Full(Node):
    shape: tuple[Node, ...]
    value: Node
    dtype: str


@dataclass(frozen=True)
class BinaryOp(Node):
    op: str
    lhs: Node
    rhs: Node


@dataclass(frozen=True)
class UnaryOp(Node):
    op: str
    operand: Node


@dataclass(frozen=True)
class Reduce(Node):
    op: str
    operand: Node
    axis: int


@dataclass(frozen=True)
class Index(Node):
    index: tuple[Node, ...]

    def __hash__(self) -> int:
        return hash(tuple(v for v in self.index))

    def __eq__(self, other) -> bool:
        return isinstance(other, Index) and self.index == other.index

    def children(self) -> list[Node]:
        return list(self.index)

    def map_children(self, f) -> Index:
        return Index(tuple(f(i) for i in self.index))


@dataclass(frozen=True)
class SymInt(Node):
    name: str

    def __eq__(self, other) -> bool:
        return isinstance(other, SymInt) and self.name == other.name

    def __floordiv__(self, other) -> Node:
        if isinstance(other, SymInt) and self.name == other.name:
            return Constant(1)
        else:
            return BinaryOp("//", self, other)

    def __hash__(self) -> int:
        return hash(self.name)

    def __mul__(self, other) -> Node:
        return BinaryOp("*", self, other)


@dataclass(frozen=True)
class Tiling:  # Not a Node
    name: str
    tensor_dims: tuple[Node, ...]
    tile_dims: tuple[Node, ...]
    dim_names: tuple[str, ...]
    order: tuple[int, ...]
    dtype: str
    paged_metadata: dict | None = field(
        default=None, compare=False, hash=False, repr=True
    )

    @property
    def num_tiles(self) -> tuple[Node, ...]:
        return tuple(
            cdiv(tensor_dim, tile_dim)
            for tensor_dim, tile_dim in zip(self.tensor_dims, self.tile_dims)
        )


@dataclass(frozen=True)
class Store(Node):
    tiling: Tiling
    index: Index
    value: Node


@dataclass(frozen=True)
class Load(Node):
    tiling: Tiling
    index: Index
    other: Constant | None


@dataclass(frozen=True)
class Assign(Node):
    target: Variable
    value: Node


@dataclass(frozen=True)
class Loop(Node):
    loop_var: Node
    start: Node
    end: Node
    body: Block


@dataclass(frozen=True)
class Where(Node):
    condition: Node
    x: Node
    y: Node


@dataclass(frozen=True)
class Arange(Node):
    start: Node
    end: Node


@dataclass(frozen=True)
class Unsqueeze(Node):
    operand: Node
    axis: int


@dataclass(frozen=True)
class Cast(Node):
    operand: Node
    dtype: str


@dataclass(frozen=True)
class CDiv(Node):
    lhs: Node
    rhs: Node


@dataclass(frozen=True)
class If(Node):
    condition: Node
    then_block: Block
    else_block: Block


@dataclass(frozen=True)
class Bool(Node):
    value: bool

    def __eq__(self, other) -> bool:
        return isinstance(other, Bool) and self.value == other.value


def cdiv(x: Node, y: Node) -> Node:
    if isinstance(x, SymInt) and isinstance(y, SymInt) and x == y:
        return Constant(1)
    elif isinstance(y, Constant) and y.value == 1:
        return x
    return CDiv(x, y)


class IRVisitor:
    def visit(self, node: Node, *args, **kwargs):
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node, *args, **kwargs)

    def generic_visit(self, node: Node, *args, **kwargs):
        raise NotImplementedError(
            f"No visit_{type(node).__name__} method for {type(node)}"
        )


class PPrintVisitor(IRVisitor):
    def __init__(self):
        self.code_lines: list[str] = []
        self.counter: int = 0
        self.visited: dict[uuid.UUID, str] = {}
        self.indent_level: int = 0

    def indented(self, s: str) -> str:
        return f"{self.indent_level * ' '}{s}"

    def get_tmp_var(self, node: Node) -> str:
        tmp_var = f"t_{self.counter}"
        self.visited[node.id] = tmp_var
        self.counter += 1
        return tmp_var

    def visit(self, node: Node):
        if node.id in self.visited:
            return self.visited[node.id]
        return super().visit(node)

    def visit_Variable(self, node: "Variable"):
        return node.name

    def visit_Constant(self, node: "Constant"):
        return str(node.value)

    def visit_BinaryOp(self, node: "BinaryOp"):
        lhs = self.visit(node.lhs)
        rhs = self.visit(node.rhs)
        tmp_var = self.get_tmp_var(node)
        self.code_lines.append(self.indented(f"{tmp_var} = {lhs} {node.op} {rhs}"))
        return tmp_var

    def visit_Index(self, node: "Index"):
        return "[" + ", ".join(self.visit(index) for index in node.index) + "]"

    def visit_Store(self, node: "Store"):
        index = self.visit(node.index)
        value = self.visit(node.value)
        self.code_lines.append(
            self.indented(f"STORE {value} to {index} in {node.tiling.name}")
        )
        return ""

    def visit_Load(self, node: "Load"):
        index = self.visit(node.index)
        tmp_var = self.get_tmp_var(node)
        other_stmt = f", other={node.other.value}" if node.other else ""
        self.code_lines.append(
            self.indented(
                f"{tmp_var} = LOAD {index} in {node.tiling.name} WITH ORDER {node.tiling.order}{other_stmt}"
            )
        )
        return tmp_var

    def visit_UnaryOp(self, node: "UnaryOp"):
        operand = self.visit(node.operand)
        tmp_var = self.get_tmp_var(node)
        self.code_lines.append(self.indented(f"{tmp_var} = {node.op} {operand}"))
        return tmp_var

    def visit_Reduce(self, node: "Reduce"):
        operand = self.visit(node.operand)
        tmp_var = self.get_tmp_var(node)
        self.code_lines.append(
            self.indented(f"{tmp_var} = {node.op}({operand}, axis={node.axis})")
        )
        return tmp_var

    def visit_Loop(self, node: "Loop"):
        loop_var = self.visit(node.loop_var)
        start = self.visit(node.start)
        end = self.visit(node.end)
        self.code_lines.append(
            self.indented(f"FOR {loop_var} IN RANGE({start}, {end})")
        )
        self.indent_level += 2
        for body_node in node.body:
            self.visit(body_node)
        self.indent_level -= 2
        return ""

    def visit_Block(self, node: "Block"):
        for body_node in node.body:
            s = self.visit(body_node)
            if s:
                self.code_lines.append(self.indented(s))
        return ""

    def visit_Assign(self, node: "Assign"):
        target = self.visit(node.target)
        value = self.visit(node.value)
        self.code_lines.append(self.indented(f"{target} = {value}"))
        return ""

    def visit_Zeros(self, node: "Zeros"):
        return f"Zeros({', '.join(self.visit(dim) for dim in node.shape)})"

    def visit_Full(self, node: "Full"):
        return f"Full({', '.join(self.visit(dim) for dim in node.shape)}, {self.visit(node.value)}, {node.dtype})"

    def visit_SymInt(self, node: "SymInt"):
        return f"SymInt({node.name})"

    def visit_Where(self, node: "Where"):
        condition = self.visit(node.condition)
        x = self.visit(node.x)
        y = self.visit(node.y)
        tmp_var = self.get_tmp_var(node)
        self.code_lines.append(
            self.indented(f"{tmp_var} = WHERE {condition}: {x} ELSE {y}")
        )
        return tmp_var

    def visit_Arange(self, node: "Arange"):
        start = self.visit(node.start)
        end = self.visit(node.end)
        tmp_var = self.get_tmp_var(node)
        self.code_lines.append(self.indented(f"{tmp_var} = ARANGE({start}, {end})"))
        return tmp_var

    def visit_Unsqueeze(self, node: "Unsqueeze"):
        operand = self.visit(node.operand)
        tmp_var = self.get_tmp_var(node)
        self.code_lines.append(
            self.indented(f"{tmp_var} = UNSQUEEZE({operand}, {node.axis})")
        )
        return tmp_var

    def visit_Cast(self, node: "Cast"):
        operand = self.visit(node.operand)
        tmp_var = self.get_tmp_var(node)
        self.code_lines.append(
            self.indented(f"{tmp_var} = CAST({operand}, {node.dtype})")
        )
        return tmp_var

    def visit_CDiv(self, node: "CDiv"):
        lhs = self.visit(node.lhs)
        rhs = self.visit(node.rhs)
        tmp_var = self.get_tmp_var(node)
        self.code_lines.append(self.indented(f"{tmp_var} = CDIV({lhs}, {rhs})"))
        return tmp_var

    def visit_If(self, node: "If"):
        condition = self.visit(node.condition)
        self.code_lines.append(self.indented(f"IF {condition}"))
        self.indent_level += 2
        for body_node in node.then_block:
            self.visit(body_node)
        self.indent_level -= 2
        return ""


def pprint_node(node: Node, print_stdout: bool = True) -> str:
    """Convenience function to print/debug the IR"""
    visitor = PPrintVisitor()
    visitor.visit(node)
    program_str = "\n".join(visitor.code_lines)
    if print_stdout:
        print(program_str)
    return program_str
