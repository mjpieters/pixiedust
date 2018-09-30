# Copyright 2018 Martijn Pieters, Zopatista Ltd.
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE.txt file for details.

import collections
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
        address = abs(address) & 0x7fffffff
        pagenum, subaddress = address // self._page_size, address % self._page_size
        self._get_page(pagenum)[subaddress] = value

    def __getitem__(self, address):
        address = abs(address) & 0x7fffffff
        pagenum, subaddress = address // self._page_size, address % self._page_size
        return self._get_page(pagenum).get(subaddress, 0)


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
        self.labels_used = {}

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
            try:
                self.validators[self.next_token()]
            except StopIteration:
                raise SyntaxError(
                    f"Missing instruction characters on line {self.pos + 1}"
                )
            if next(self.tokens, None) is not None:
                raise SyntaxError(f"Trailing characters on line {self.pos + 1}")
        if self.labels_used.keys() > self.labels.keys():
            # jump to non-existing label
            unavailable = self.labels_used.keys() - self.labels
            lineno = min(self.labels_used[l] for l in unavailable)
            raise SyntaxError(f"Invalid label target on line {lineno}")

    def execute_next(self):
        instruction = self.instructions[self.pos]
        self.tokens = iter(tokenizer(instruction))
        self.next_token = partial(next, self.tokens)
        self.opcodes[self.next_token()]
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
            bits = "".join(takewhile(lambda t: t != "*", islice(self.tokens, 33)))
            neg = len(bits) > 31 and bits[0] == "+"
            return int(bits[-31:].translate(_b) or "0", 2) - (0x80000000 if neg else 0)

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

    def expression(self):
        return self[self.next_token() + self.next_token()]

    def validate_register(self, toset=False):
        register = self.next_token() + self.next_token()
        if register == ".*":
            if toset:
                # reserved for future use
                raise SyntaxError(f"No such register: .*, on line {self.pos + 1}")
            else:
                # consume the literal tokens
                bits = "".join(takewhile(lambda t: t != "*", islice(self.tokens, 33)))
                if len(bits) >= 33:
                    # too many bits
                    raise SyntaxError(f"Invalid number literal on line {self.pos + 1}")

    def validate_expression(self):
        self.validate_register()

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

    @op_math_copy.validator
    def op_math_copy(self):
        self.validate_register(True)
        self.validate_expression()

    @opcode("*+")
    def op_math_add_sub(self, _o={"+": operator.add, ".": operator.sub}):  # noqa B006
        """* O: ++ for addition, +. for subtraction"""
        oper = _o[self.next_token()]
        register = self.next_token() + self.next_token()
        x = self.expression()
        y = self.expression()
        self[register] = oper(x, y)

    @op_math_add_sub.validator
    def op_math_add_sub(self):
        oper = self.next_token()
        if oper == "*":
            # *+* is reserved for future use.
            raise SyntaxError(f"No such math operator: *+*, on line {self.pos + 1}")
        self.validate_register(True)
        self.validate_expression()
        self.validate_expression()

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

    @op_math_mul_div_mod.validator
    def op_math_mul_div_mod(self):
        self.next_token()  # skip operator token
        self.validate_register(True)
        self.validate_expression()
        self.validate_expression()

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

    @op_comp.validator
    def op_comp(self):
        self.next_token()  # skip comparator token
        self.validate_expression()
        self.validate_expression()

    @opcode("++")
    def op_print(self):
        """++ X prints the Unicode character represented by expression X to STDOUT."""
        x = self.expression()
        self.stdout.write(chr(x))

    @op_print.validator
    def op_print(self):
        self.validate_expression()

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

    @op_jump_label.validator
    def op_jump_label(self):
        """Validate the label exists during pre-pass"""
        self.next_token()  # the test token
        label = "".join(self.tokens)
        self.labels_used[label] = self.pos + 1


if __name__ == "__main__":
    duster = PixieDust()
    with open(sys.argv[1], "r") as instructions:
        duster.execute(instructions.read())
