# Copyright 2018 Martijn Pieters, Zopatista Ltd.
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE.txt file for details.

import collections
import itertools
import operator
import re
import sqlite3
import struct
import sys

from functools import partial
from itertools import islice, takewhile


illegal = re.compile(r"[^*+.\s]").search
tokenizer = re.compile(r"[*+.]").findall


class opcode:
    """Descriptor / decorator for operator methods"""

    def __init__(self, tokens, fget=None, fvalidator=None):
        self.tokens = tokens
        self.fget = fget
        self.fvalidator = fvalidator

    def __set_name__(self, owner, name):
        owner.opcodes[self.tokens] = self.fget

    def __call__(self, fget):
        return type(self)(self.tokens, fget, self.fvalidator)

    def validator(self, fvalidator):
        return type(self)(self.tokens, self.fget, fvalidator)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self.fget.__get__(instance, owner)


class Opcodes(dict):
    """Operator registry

    On first access on an instance, the registered opcode functions are bound
    to the instance the result is cached. Opcodes are executed on access.

    """

    def __init__(self, items=(), name=None, instance=None):
        self.name = name
        self.instance = instance
        if instance is not None:
            # bind everything to instance just once
            items = ((t, o.__get__(instance)) for t, o in items)
        super().__init__(items)

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        # cache on get, to get out of the way next time
        bound = instance.__dict__[self.name] = Opcodes(
            self.items(), name=self.name, instance=instance
        )
        return bound

    def __getitem__(self, opcode):
        return super().__getitem__(opcode)()

    def __missing__(self, opcode):
        """Handle intermediate opcodes

        For opcodes like ++, the + prefix is not registered
        but this handler creates one which passes on the call
        to the composite token.

        """
        # Potentially this could lead to handlers being generated
        # for non-existing tokens, but instruction lines are
        # not infinite, so it'll be fine.
        map = getattr(self.instance, self.name)

        def intermediate(self):
            return map[opcode + self.next_token()]

        bound = self[opcode] = intermediate.__get__(self.instance)
        return bound


class SQLiteMemory:
    """31-bit addressable memory for PixieDust programs

    Memory cells are 4-byte words, containing signed integers.

    """

    # memory is swapped out to a sqlite table! Because I don't want to think
    # about someone actually using the full address space, sqlite3 would
    # neatly handle this by swapping to temp disk space instead.
    # default page size gives us pages with ~128MB of Python integer storage,
    # and there are 1024 pages. With 32 pages active and full, about 4GB
    # of Python heap memory is used.
    def __init__(self, page_size=2 ** 21, max_active=32):
        self._page_size = page_size
        self._max_active = max_active
        self._pages = collections.OrderedDict()
        self._conn = sqlite3.connect("")  # sqlite opens a temp file for this
        self._conn.execute(
            """
            CREATE TABLE memory (
                page INT, address INT, value INT,
                PRIMARY KEY(page, address)
            )
            """
        )
        self._cursor = self._conn.cursor()

    def _get_page(self, pagenum):
        if pagenum in self._pages:
            self._pages.move_to_end(pagenum)
            return self._pages[pagenum]

        with self._conn:
            self._pages[pagenum] = page = dict(
                self._cursor.execute(
                    """
                    SELECT address, value FROM memory
                    WHERE page = ?
                    """,
                    (pagenum,),
                )
            )

        self._maybe_evict()
        return page

    def _maybe_evict(self):
        if len(self._pages) > self._max_active:
            pagenum, page = self._pages.popitem(last=False)
            with self._conn:
                self._cursor.executemany(
                    f"""
                    INSERT OR REPLACE INTO memory (page, address, value)
                    VALUES ({pagenum}, ?, ?)
                    """,
                    page.items(),
                )

    def __setitem__(self, address, value):
        address = address & 0x7FFFFFFF
        pagenum, subaddress = address // self._page_size, address % self._page_size
        self._get_page(pagenum)[subaddress] = value

    def __getitem__(self, address):
        address = address & 0x7FFFFFFF
        pagenum, subaddress = address // self._page_size, address % self._page_size
        return self._get_page(pagenum).get(subaddress, 0)


# mapping pixiedust characters to bits for the .* literal syntax
_dustbin_map = str.maketrans(".+", "01")
# label offset dummy


def _offset_missing():
    raise RuntimeError("Missing label offset identity callable")


_offset_placeholder = _offset_missing, 0


# handle casting signed integers by packing into to 8 long long bytes, then
# slicing back target size. with 8 bytes we can handle any overflow scenario.
# You can't use masking as that casts to an unsigned int instead.
signed32bit = (
    (partial(struct.pack, "!q"), 1),
    (operator.itemgetter(slice(-4, None)), 1),
    (partial(struct.unpack, "!l"), 1),
    (operator.itemgetter(0), 1),
)


class PixieDust:
    # PixieDust interpreter. Compiles instructions to a tuple of
    # (callable, argcount) operations each. Results are pushed on
    # a stack, and callables are passed argcount top values from
    # the stack.

    opcodes = Opcodes()

    def __init__(self, stdout=sys.stdout, stdin=sys.stdin):
        self.registers = {}
        self.memory = SQLiteMemory()
        self.stdout = stdout
        self.stdin = stdin

    # program execution
    def execute(self, dust):
        instructions = self.compile(dust)
        self.pos = 0
        while 0 <= self.pos < len(instructions):
            # An instruction consists of (callable, argcount) entries,
            # where argcount is passed the most recent argcount of results,
            # in stack order (top-most first)
            stack = collections.deque()
            for op, count in instructions[self.pos]:
                args = (stack.pop() for _ in itertools.repeat(None, count))
                stack.append(op(*args))
            self.pos += 1

    def compile(self, dust):
        """Convert instructions to a series of (operation, argcount) sequences"""
        self.labels = {}
        self.label_jumps = {}
        compiled = []

        for i, instruction in enumerate(dust.splitlines()):
            self.pos = i
            if illegal(instruction):
                raise SyntaxError(f"Invalid characters on line {self.pos + 1}")
            self.tokens = iter(tokenizer(instruction))
            self.next_token = partial(next, self.tokens)
            try:
                compiled.append(self.opcodes[self.next_token()])
            except StopIteration:
                raise SyntaxError(
                    f"Missing instruction characters on line {self.pos + 1}"
                )
            if next(self.tokens, None) is not None:
                raise SyntaxError(f"Trailing characters on line {self.pos + 1}")

        # set jump offsets, needs to be done at the end when all label targets
        # have been processed.
        for label, positions in self.label_jumps.items():
            try:
                target = self.labels[label]
            except KeyError:
                # jump to non-existing label
                raise SyntaxError(f"Invalid label target on line {positions[0] + 1}")
            # replace offset placeholder with actual relative offset
            for pos in positions:
                assert compiled[pos][0] is _offset_placeholder
                offset_op = partial(int, target - pos), 0
                compiled[pos] = (offset_op, *compiled[pos][1:])

        return compiled

    # register handling

    def compile_register_set(self, register=None):
        """Return operations that sets the register to a value on the stack"""
        if register is None:
            register = self.next_token() + self.next_token()
        if register not in {"*.", "*+", ".*"}:
            return (
                *signed32bit,
                (partial(operator.setitem, self.registers, register), 1),
            )
        elif register == "*+":  # value as Unicode char to stdout
            # mask the integer value and convert to a unicode character first.
            # this should be a Java (char) 16 bit range, not full Unicode
            return (
                (partial(operator.and_, 0xFFFF), 1),
                (chr, 1),
                (partial(self.stdout.write), 1),
            )
        elif register == "*.":  # memory access
            # fetch the ** register first, then set the memory value with that result
            rget = partial(self.registers.get, "**", 0), 0
            mset = partial(operator.setitem, self.memory), 2
            return (*signed32bit, rget, mset)
        # reserved for future use
        raise SyntaxError(f"No such register: {register}, on line {self.pos + 1}")

    def compile_register_get(self, register=None, _b=_dustbin_map):
        """Return operations that produce the register value"""
        if register is None:
            register = self.next_token() + self.next_token()
        if register not in {"*.", "*+", ".*"}:
            return ((partial(self.registers.get, register, 0), 0),)
        elif register == "*+":  # read a unicode character from stdin
            # convert the character read to a 16-bit signed integer
            return (
                (partial(self.stdin.read, 1), 0),
                (ord, 1),
                (partial(operator.and_, 0xFFFF), 1),
            )
        elif register == "*.":  # memory access
            # fetch the ** register first, then fetch the memory value with that result
            rget = partial(self.registers.get, "**", 0), 0
            mget = partial(operator.getitem, self.memory), 1
            return rget, mget
        elif register == ".*":  # literal value
            # consume the literal tokens
            bits = "".join(takewhile(lambda t: t != "*", islice(self.tokens, 33)))
            if len(bits) >= 33:
                # too many bits
                raise SyntaxError(f"Invalid number literal on line {self.pos + 1}")
            neg = len(bits) > 31 and bits[0] == "+"
            value = int(bits[-31:].translate(_b) or "0", 2) - (0x80000000 if neg else 0)
            return ((partial(int, value), 0),)

    # opcode implementation

    # * O R X Y is a mathematical operation
    #
    # O specifies the operation to use: ...
    # R specifies the register to store the result to.
    # X and Y are expressions.

    @opcode("*.")
    def op_math_copy(self):
        """* O: . for copy

        For a copy operation, Y should be omitted.

        """
        register_set = self.compile_register_set()
        x_get = self.compile_register_get()
        return (*x_get, *register_set)

    @opcode("*+")
    def op_math_add_sub(self, _o={"+": operator.add, ".": operator.sub}):  # noqa B006
        """* O: ++ for addition, +. for subtraction"""
        try:
            oper = _o[self.next_token()], 2
        except KeyError as e:
            # *+* is reserved for future use.
            raise SyntaxError(
                f"No such math operator: *+{e.args[0]}, on line {self.pos + 1}"
            )
        register_set = self.compile_register_set()
        x_get = self.compile_register_get()
        y_get = self.compile_register_get()
        return (*y_get, *x_get, oper, *register_set)

    @opcode("**")
    def op_math_mul_div_mod(
        self,
        _o={"*": operator.mul, ".": operator.floordiv, "+": operator.mod},  # noqa B006
    ):
        """* O: ** for multiplication, *. for division, *+ for modulo"""
        oper = _o[self.next_token()], 2
        register_set = self.compile_register_set()
        x_get = self.compile_register_get()
        y_get = self.compile_register_get()
        # put y on the stack first
        return (*y_get, *x_get, oper, *register_set)

    @opcode(".")
    def op_comp(
        self, _c={"*": operator.eq, "+": operator.lt, ".": operator.gt}  # noqa B006
    ):
        """. C X Y performs the comparison specified by C

        ... and stores it with 0/1 in the .. register

        =<> are indicated by *+., respectively. X and Y are expressions.

        """
        comp = (_c[self.next_token()], 2), (int, 1)
        x_get = self.compile_register_get()
        y_get = self.compile_register_get()
        register_set = self.compile_register_set("..")
        return (*y_get, *x_get, *comp, *register_set)

    @opcode("++")
    def op_print(self):
        """++ X prints the Unicode character represented by expression X to STDOUT."""
        x_get = self.compile_register_get()
        # mask to Java char range
        print_ops = self.compile_register_set("*+")
        return (*x_get, *print_ops)

    @opcode("+.")
    def op_set_label(self):
        """+. L defines a program label; L can be any number of characters."""
        label = "".join(self.tokens)
        if label in self.labels:
            raise SyntaxError(
                f"Re-definition of label {label!r} on line {self.pos + 1}"
            )
        self.labels[label] = self.pos
        return ()  # return noop to preserve instruction positions

    @opcode("+*")
    def op_jump_label(self, _t={"*": operator.truth, ".": operator.not_}):  # noqa B006
        """+* T L jumps to label L based on the condition T.

        T can be
            * to jump if .. is not 0,
            . to jump if .. is 0, or
            + to jump regardless of the value in ...
        """
        try:
            test_op = _t[self.next_token()], 1
        except KeyError:
            # jump unconditional, no test and adjustment needed
            test_ops = ()
        else:
            register_get = self.compile_register_get("..")
            # Take the test output (True or False) and multiply this with the offset
            # The result is either the offset, or 0
            adjust_offset_ops = operator.mul, 2
            test_ops = (*register_get, test_op, adjust_offset_ops)

        # register the target for the compiler to later on insert
        # an offset into the operations
        label = "".join(self.tokens)
        self.label_jumps.setdefault(label, []).append(self.pos)

        # add the (updated) offset to self.pos
        update_pos_op = (
            (partial(getattr, self, "pos"), 0),
            (operator.add, 2),
            (partial(setattr, self, "pos"), 1),
        )
        return (_offset_placeholder, *test_ops, *update_pos_op)


if __name__ == "__main__":
    duster = PixieDust()
    with open(sys.argv[1], "r") as instructions:
        duster.execute(instructions.read())
