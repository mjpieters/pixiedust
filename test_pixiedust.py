# Copyright 2018 Martijn Pieters, Zopatista Ltd.
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE.txt file for details.

import io
import random
import unittest

import pixiedust


class OpcodeTests(unittest.TestCase):
    def test_decorator_registers(self):
        class Foo:
            opcodes = {}

            @pixiedust.opcode("b")
            def bar(self):
                return "bar"

        self.assertTrue("b" in Foo.opcodes)
        self.assertIs(Foo.opcodes["b"], Foo.bar.fget)

    def test_opcode_descriptor_binds(self):
        class Foo:
            opcodes = {}

            @pixiedust.opcode("bar")
            def bar(self):
                pass

        with self.subTest("unbound"):
            # class access does not bind (return self)
            unbound = Foo.bar
            self.assertIs(unbound, Foo.__dict__["bar"])

        with self.subTest("bound"):
            # instance access does bind, producing the bound method
            # that was decorated.
            instance = Foo()
            bound = instance.bar
            self.assertIs(bound.__func__, Foo.__dict__["bar"].fget)


class OpcodesTests(unittest.TestCase):
    def test_opcodes_descriptor_binds(self):
        class Foo:
            registry = pixiedust.Opcodes()

            def bar(self):
                pass

            # manual registration of a descriptor object
            registry["b"] = bar

        with self.subTest("unbound"):
            # class access does not bind (return self)
            unbound = Foo.registry
            self.assertIs(unbound, Foo.__dict__["registry"])

            # values are unchanged, using dict.get to avoid __getitem__ hook
            self.assertIs(unbound.get("b"), Foo.bar)

        with self.subTest("bound"):
            # instance access binds and caches
            instance = Foo()
            bound = instance.registry
            self.assertIsNot(bound, unbound)
            self.assertIsInstance(bound, pixiedust.Opcodes)
            self.assertIn("registry", instance.__dict__)

            # the values have all been bound
            # using dict.get to avoid __getitem__
            self.assertIs(bound.get("b").__func__, instance.bar.__func__)

    def test_auto_intermediaries(self):
        class Foo:
            registry = pixiedust.Opcodes()

            def test_tokens(self, tokens):
                tokeniter = iter(tokens)
                self.next_token = lambda: next(tokeniter)
                self.called = None
                self.registry[self.next_token()]
                return self.called

            def bar(self):
                self.called = "bar"

            def baz(self):
                self.called = "baz"

            # manual registrations for this test
            registry["abr"] = bar
            registry["abz"] = baz

        # we don't care about the unbound class.opcodes[...] case,
        # it's the full 'abc' path that must work,
        instance = Foo()
        self.assertEqual(instance.test_tokens("abr"), "bar")
        self.assertEqual(instance.test_tokens("abz"), "baz")
        # intermediares are only generated on first access
        self.assertIn("a", instance.registry)
        self.assertIn("ab", instance.registry)

    def test_opcodes_get_calls(self):
        class Foo:
            opcodes = pixiedust.Opcodes()

            def bar(self):
                return "bar called"

            # manual registration of a descriptor object
            opcodes["bar"] = bar

        # we don't care about the unbound class.opcodes[...] case
        instance = Foo()
        bound = instance.opcodes
        self.assertEqual(bound["bar"], "bar called")


class SQLMemoryTests(unittest.TestCase):
    def test_basic_get_set(self):
        memory = pixiedust.SQLiteMemory()

        address = random.randint(0, 0x7FFFFFFF)
        value = random.randint(0, 0xFFFFFFFF)

        memory[address] = value
        self.assertEqual(memory[address], value)

    def test_boundaries(self):
        # exceeding the value and address boundaries are fine
        # as both are masked.
        memory = pixiedust.SQLiteMemory()

        address_boundaries = 0, 0x7FFFFFFF
        value_boundaries = -0x80000000, 0x7FFFFFFF

        address = random.randint(*address_boundaries)
        value = random.randint(*value_boundaries)

        targets = {
            "address": (address_boundaries, value),
            "value": (value_boundaries, address),
        }

        for target, (boundaries, testint) in targets.items():
            with self.subTest(target):
                subtests = {
                    "at boundaries": boundaries,
                    "underflow": (boundaries[0] - random.randint(1, 1000),),
                    "overflow": (boundaries[1] + random.randint(1, 1000),),
                }
                for subtest, testvalues in subtests.items():
                    with self.subTest(subtest):
                        for totest in testvalues:
                            if target == "address":
                                address, value = totest, testint
                            else:
                                address, value = testint, totest
                            memory[address] = value
                            self.assertEqual(memory[address], value)


class PixieDustTests(unittest.TestCase):
    def test_helloworld(self):
        helloworld = (  # original hello world golf sample
            # Hello
            "++.*+..+...\n++.*++..+.+\n++.*++.++..\n++.*++.++..\n++.*++.++++\n"
            # ,<space>
            "++.*+.++..\n++.*+.....\n"
            # World
            "++.*+.+.+++\n++.*++.++++\n++.*+++..+.\n++.*++.++..\n++.*++..+..\n"
            # !
            "++.*+....+"
        )
        out = io.StringIO()
        interpreter = pixiedust.PixieDust(stdout=out)
        interpreter.execute(helloworld)
        self.assertEqual(out.getvalue(), "Hello, World!")

    def test_expressions(self):
        bindust_map = str.maketrans("01", ".+")
        pos_increment = "*++ ** ** .* +\n"  # add 1 to the ** register
        # try several different literal sizes
        for i in range(0, 0x80000000, 0x1234567):
            long_literal = format(i, "032b").translate(bindust_map) + "*"
            negative_literal = "+" + long_literal[1:]
            short_literal = long_literal.lstrip(".").rstrip("*")

            interpreter = pixiedust.PixieDust()
            interpreter.execute(
                # store each of the literals in memory
                # *. *. copies to the memory pointer, *++ ** + increments the pointer
                pos_increment.join(
                    [
                        f"*. *. .* {long_literal}\n",
                        f"*. *. .* {short_literal}\n",
                        f"*. *. .* {negative_literal}\n",
                    ]
                )
            )
            self.assertEqual(interpreter.memory[0], i)
            self.assertEqual(interpreter.memory[1], i)
            self.assertEqual(interpreter.memory[2], -0x80000000 + i)

    def test_output_unicode(self):
        with self.subTest("print"):
            out = io.StringIO()
            interpreter = pixiedust.PixieDust(stdout=out)
            # print U+2728 SPARKLES
            interpreter.execute("++ .* +..+++..+.+...\n")
            self.assertEqual(out.getvalue(), "\u2728")

        with self.subTest("*+ register"):
            out = io.StringIO()
            interpreter = pixiedust.PixieDust(stdout=out)
            # copy U+2728 SPARKLES to the *+ register
            interpreter.execute("*. *+ .* +..+++..+.+...\n")
            self.assertEqual(out.getvalue(), "\u2728")

    def test_input_unicode(self):
        in_ = io.StringIO("\u2728")
        interpreter = pixiedust.PixieDust(stdin=in_)
        # copy byte from stdin to memory address 0
        interpreter.execute("*. *. *+\n")
        self.assertEqual(interpreter.memory[0], 0x2728)

    def test_jump_unconditional(self):
        interpreter = pixiedust.PixieDust()
        interpreter.execute(
            # set memory address 0 to 42
            "*. *. .* +.+.+.\n"
            # unconditial jump to *..*
            "+* + *..*\n"
            # set memory address 0 to 81
            "*. *. .* +.+...+\n"
            # set label *..*; code should jump here
            "+. *..*\n"
        )
        self.assertEqual(interpreter.memory[0], 42)

    def test_jump_nonzero(self):
        interpreter = pixiedust.PixieDust()
        interpreter.execute(
            # set memory address 0 to 81
            "*. *. .* +.+...+\n"
            # set .. register to 0 (the default)
            "*. .. .* .\n"
            # if .. is 1, jump to *..*
            "+* * *..*\n"
            # set memory address 0 to 42
            "*. *. .* +.+.+.\n"
            # set .. register to 1
            "*. .. .* +\n"
            # if .. is 1, jump to *..*
            "+* * *..*\n"
            # set memory address 0 to 81
            "*. *. .* +.+...+\n"
            # set label *..*; code should jump here
            "+. *..*\n"
        )
        self.assertEqual(interpreter.memory[0], 42)

    def test_jump_zero(self):
        interpreter = pixiedust.PixieDust()
        interpreter.execute(
            # set memory address 0 to 81
            "*. *. .* +.+...+\n"
            # set .. register to 1
            "*. .. .* +\n"
            # if .. is 0, jump to *..*
            "+* . *..*\n"
            # set memory address 0 to 42
            "*. *. .* +.+.+.\n"
            # set .. register to 0
            "*. .. .* .\n"
            # if .. is 0, jump to *..*
            "+* . *..*\n"
            # set memory address 0 to 81
            "*. *. .* +.+...+\n"
            # set label *..*; code should jump here
            "+. *..*\n"
        )
        self.assertEqual(interpreter.memory[0], 42)

    def test_math(self):
        interpreter = pixiedust.PixieDust()

        # tests set memory address 0 to the result
        # of the operation on two literals.
        tests = {
            "add": "*++ *. .* +.++.* .* +.+..\n",  # 22 + 20
            "subtract": "*+. *. .* +++++.* .* +.+..\n",  # 62 - 20
            "multiply": "*** *. .* +.+.+* .* +.\n",  # 21 * 2
            "divide": "**. *. .* +++++++* .* ++\n",  # 127 // 3
            "modulo": "**+ *. .* +++........+.* .* +.+...+\n",  # 7170 % 81
        }

        for name, dust in tests.items():
            with self.subTest(name):
                interpreter.execute(dust)
                self.assertEqual(interpreter.memory[0], 42)

    def test_overflow(self):
        interpreter = pixiedust.PixieDust()

        with self.subTest("memory set"):
            # 0x7FFFFFFF + 1
            interpreter.execute("*++ *. .* .+++++++++++++++++++++++++++++++* .* +\n")
            self.assertEqual(interpreter.memory[0], -0x80000000)

        with self.subTest("register set"):
            # 0x7FFFFFFF + 1
            interpreter.execute("*++ .+ .* .+++++++++++++++++++++++++++++++* .* +\n")
            self.assertEqual(interpreter.registers.get(".+"), -0x80000000)

        with self.subTest("write stdout"):
            out = io.StringIO()
            interpreter = pixiedust.PixieDust(stdout=out)
            # copy -1 literal into the *+ registry
            interpreter.execute("*. *+ .* ++++++++++++++++++++++++++++++++\n")
            self.assertEqual(out.getvalue(), "\uFFFF")

        with self.subTest("read stdin"):
            in_ = io.StringIO("\U0001F9DA")  # U+1F9DA FAIRY
            interpreter = pixiedust.PixieDust(stdin=in_)
            # copy stdin character into memory
            interpreter.execute("*. *. *+\n")
            self.assertEqual(interpreter.memory[0], 0xF9DA)

    def test_comp(self):
        interpreter = pixiedust.PixieDust()

        tests = {
            "equals": (".* .* +* .* +\n", ".* .* +* .* .\n"),
            "lower": (".+ .* .* .* +\n", ".+ .* .* .* .\n"),
            "greater": (".. .* +* .* .\n", ".. .* .* .* .\n"),
        }
        for name, (true, false) in tests.items():
            with self.subTest(name):
                interpreter.execute(true)
                self.assertEqual(interpreter.registers[".."], 1)
                interpreter.execute(false)
                self.assertEqual(interpreter.registers[".."], 0)


class PixieDustSyntaxErrorTests(unittest.TestCase):
    def test_label_errors(self):
        interpreter = pixiedust.PixieDust()

        with self.subTest("redefinition"):
            good_label = "".join([random.choice("*+.") for _ in range(10)])
            with self.assertRaisesRegex(
                SyntaxError, r"Re-definition of label '[*+.]+' on line 2"
            ):
                interpreter.execute(
                    # set good_label
                    f"+. {good_label}\n"
                    # set it again
                    f"+. {good_label}\n"
                )

        with self.subTest("invalid target"):
            bad_label = "".join([random.choice("*+.") for _ in range(10)])
            with self.assertRaisesRegex(SyntaxError, r"Invalid label target on line 1"):
                interpreter.execute(
                    # jump to bad label
                    f"+* + {bad_label}\n"
                )

        with self.subTest("invalid target after valid"):
            with self.assertRaisesRegex(SyntaxError, r"Invalid label target on line 3"):
                interpreter.execute(
                    # set good_label
                    f"+. {good_label}\n"
                    # jnz to good label
                    f"+* * {good_label}\n"
                    # jump to bad label
                    f"+* + {bad_label}\n"
                )

    def test_trailing(self):
        interpreter = pixiedust.PixieDust()

        with self.assertRaisesRegex(SyntaxError, r"Trailing characters on line 2"):
            interpreter.execute(
                # copy 0 literal to the ** register
                "*. ** .* .*\n"
                # copy 0 literal to the ** register, with extra dots
                "*. ** .* .* ....\n"
            )

    def test_invalid_chars(self):
        interpreter = pixiedust.PixieDust()

        with self.assertRaisesRegex(SyntaxError, r"Invalid characters on line 2"):
            interpreter.execute(
                # copy 0 literal to the ** register
                "*. ** .* .*\n"
                # copy 0 literal to the ** register, with extra dots
                "*. ** .* .* SPAM\n"
            )

    def test_missing_chars(self):
        interpreter = pixiedust.PixieDust()

        with self.assertRaisesRegex(
            SyntaxError, r"Missing instruction characters on line 2"
        ):
            interpreter.execute(
                # copy 0 literal to the ** register
                "*. ** .* .*\n"
                # copy to the ** register, incomplete expression
                "*. ** .\n"
            )

    def test_setting_literal_register(self):
        interpreter = pixiedust.PixieDust()

        with self.assertRaisesRegex(SyntaxError, r"No such register: \.\*, on line 2"):
            interpreter.execute(
                # copy 0 literal to the ** register
                "*. ** .* .*\n"
                # copy 0 literal to the .* register (the literal register)
                "*. .* .* .*\n"
            )

    def test_invalid_math_operator(self):
        interpreter = pixiedust.PixieDust()

        with self.assertRaisesRegex(
            SyntaxError, r"No such math operator: \*\+\*, on line 2"
        ):
            interpreter.execute(
                # copy 0 literal to the ** register
                "*. ** .* .*\n"
                # math operator * on two 0 literals
                "*+ * .* .* .* .*\n"
            )

    def test_invalid_literal(self):
        interpreter = pixiedust.PixieDust()

        with self.assertRaisesRegex(SyntaxError, r"Invalid number literal on line 2"):
            interpreter.execute(
                # copy 0 literal to the ** register
                "*. ** .* .*\n"
                # copy 0 literal to the ** register, but with a bit too many
                "*. ** .* .................................*\n"
            )


if __name__ == "__main__":
    unittest.main()
