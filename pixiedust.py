# Copyright 2018 Martijn Pieters, Zopatista Ltd.
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE.txt file for details.

import operator
import re
import sqlite3
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
        owner.validators[self.tokens] = self.fvalidator or (lambda *args: None)

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
            map[opcode + self.next_token()]

        bound = self[opcode] = intermediate.__get__(self.instance)
        return bound


class SQLiteMemory:
    """31-bit addressable memory for PixieDust programs

    Memory cells are 4-byte words, containing signed integers.

    """

    # memory is a sqlite table! Because I don't want to think about someone
    # actually using the full address space, sqlite3 would neatly handle this
    # by swapping to temp disk space instead.
    def __init__(self):
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute("CREATE TABLE memory (address INT PRIMARY KEY, value INT)")
        self._cursor = self._conn.cursor()

    def __setitem__(self, address, value):
        with self._conn:
            self._cursor.execute(
                """
                INSERT OR REPLACE INTO memory (address, value)
                VALUES (?, ?)""",
                (abs(address) & 0x7fffffff, value),
            )

    def __getitem__(self, address):
        return self._cursor.execute(
            """
            WITH addresses(address) AS (SELECT ?)
            SELECT IFNULL(value, 0) as value
            FROM addresses
            LEFT JOIN "memory" USING (address);
            """,
            (abs(address) & 0x7fffffff,),
        ).fetchone()[0]


# mapping pixiedust characters to bits for the .* literal syntax
_dustbin_map = str.maketrans(".+", "01")


class PixieDust:
    validators = Opcodes()
    opcodes = Opcodes()

    def __init__(self, stdout=sys.stdout, stdin=sys.stdin):
        self.registers = {}
        self.memory = SQLiteMemory()
        self.stdout = stdout
        self.stdin = stdin
        self.instructions = []
        self.labels = {}
        # used by pre-pass validation
        self.labels_used = set()

    # program execution
    def execute(self, dust):
        self.instructions = dust.splitlines()
        self.pre_pass()
        self.pos = 0
        while 0 <= self.pos < len(self.instructions):
            self.execute_next()

    def pre_pass(self):
        """Check instructions, parse out labels"""
        for i, instruction in enumerate(self.instructions):
            self.pos = i
            if illegal(instruction):
                raise SyntaxError(f"Invalid characters on line {self.pos + 1}")
            self.tokens = iter(tokenizer(instruction))
            self.next_token = partial(next, self.tokens)
            self.validators[self.next_token()]

    def execute_next(self):
        instruction = self.instructions[self.pos]
        self.tokens = iter(tokenizer(instruction))
        self.next_token = partial(next, self.tokens)
        self.opcodes[self.next_token()]
        if next(self.tokens, None) is not None:
            raise SyntaxError(f"Trailing characters on line {self.pos + 1}")
        self.pos += 1

    # register handling
    def __getitem__(self, register, _b=_dustbin_map):
        assert len(register) == 2
        if register not in {"*.", "*+", ".*"}:
            return self.registers.get(register, 0)
        if register == "*.":
            return self.memory[self["**"]]
        elif register == "*+":
            return self.stdin.read(1)
        elif register == ".*":
            # up to 32 bits, terminated by * or the end of the instruction
            # integers are signed 32-bit values
            bits = "".join(takewhile(lambda t: t != "*", islice(self.tokens, 32)))
            neg = len(bits) > 31 and bits[0] == "+"
            return int(bits[-31:].translate(_b), 2) - (0x80000000 if neg else 0)

    def __setitem__(self, register, value):
        assert len(register) == 2
        # masking unsigned integers is .. hard.
        value = value & 0x7fffffff if value >= 0 else -((-value - 1) & 0x7fffffff) - 1
        if register not in {"*.", "*+", ".*"}:
            self.registers[register] = value
            return
        if register == "*.":
            self.memory[self["**"]] = value
        elif register == "*+":
            self.stdout.write(value)
        elif register == ".*":
            # reserved for future use
            raise SyntaxError(f"No such register: .*, on line {self.pos + 1}")

    def expression(self):
        return self[self.next_token() + self.next_token()]

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
        register = self.next_token() + self.next_token()
        x = self.expression()
        self[register] = x

    @opcode("*+")
    def op_math_add_sub(self, _o={"+": operator.add, ".": operator.sub}):  # noqa B006
        """* O: ++ for addition, +. for subtraction"""
        try:
            oper = _o[self.next_token()]
        except KeyError:
            # *+* is reserved for future use.
            raise SyntaxError(f"No such opcode: *+*, on line {self.pos + 1}")
        register = self.next_token() + self.next_token()
        x = self.expression()
        y = self.expression()
        self[register] = oper(x, y)

    @opcode("**")
    def op_math_mul_div_mod(
        self,
        _o={"*": operator.mul, ".": operator.floordiv, "+": operator.mod},  # noqa B006
    ):
        """* O: ** for multiplication, *. for division, *+ for modulo"""
        oper = _o[self.next_token()]
        register = self.next_token() + self.next_token()
        x = self.expression()
        y = self.expression()
        self[register] = oper(x, y)

    @opcode(".")
    def op_comp(
        self, _c={"*": operator.eq, "+": operator.lt, ".": operator.gt}  # noqa B006
    ):
        """. C X Y performs the comparison specified by C

        ... and stores it with 0/1 in the .. register

        =<> are indicated by *+., respectively. X and Y are expressions.

        """
        comp = _c[self.next_token()]
        x = self.expression()
        y = self.expression()
        self[".."] = int(comp(x, y))

    @opcode("++")
    def op_print(self):
        """++ X prints the Unicode character represented by expression X to STDOUT."""
        x = self.expression()
        self.stdout.write(chr(x))

    @opcode("+.")
    def op_set_label(self):
        """+. L defines a program label; L can be any number of characters."""
        # Label is set in the pre-pass phase

    @op_set_label.validator
    def op_set_label(self):
        """Set the label during pre-pass"""
        label = "".join(self.tokens)
        self.labels[label] = self.pos

    @opcode("+*")
    def op_jump_label(
        self, _t={"*": operator.truth, ".": operator.not_, "+": str}  # noqa B006
    ):
        """+* T L jumps to label L based on the condition T.

        T can be
            * to jump if .. is not 0,
            . to jump if .. is 0, or
            + to jump regardless of the value in ...
        """
        # evil grin: `str(0)` and `str(1)` are both true values.
        # So for +*+ (unconditional jump, `if str(register value):` will always be true
        test = _t[self.next_token()]
        label = "".join(self.tokens)
        if test[self[".."]]:
            self.pos = self.labels[label]


if __name__ == "__main__":
    duster = PixieDust()
    with open(sys.argv[1], "r") as instructions:
        duster.execute(instructions.read())
