# Copyright 2018 Martijn Pieters, Zopatista Ltd.
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE.txt file for details.

import io
import random
import unittest

import pixiedust


class OpcodeTests(unittest.TestCase):
    def test_decorator_registers_without_validator(self):
        class Foo:
            opcodes = {}
            validators = {}

            @pixiedust.opcode("b")
            def bar(self):
                return "bar"

        self.assertTrue("b" in Foo.opcodes)
        self.assertTrue("b" in Foo.validators)
        self.assertIs(Foo.opcodes["b"], Foo.bar.fget)
        # validator is a noop lambda
        self.assertEqual(Foo.validators["b"].__name__, "<lambda>")

    def test_decorator_registers_with_validator(self):
        class Foo:
            opcodes = {}
            validators = {}

            @pixiedust.opcode("b")
            def bar(self):
                return "bar"

            @bar.validator
            def bar(self):
                return "validator"

        self.assertTrue("b" in Foo.opcodes)
        self.assertTrue("b" in Foo.validators)
        self.assertIs(Foo.opcodes["b"], Foo.bar.fget)
        self.assertIs(Foo.validators["b"], Foo.bar.fvalidator)

    def test_opcode_descriptor_binds(self):
        class Foo:
            opcodes = {}
            validators = {}

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

        address = random.randint(0, 0x7fffffff)
        value = random.randint(0, 0xffffffff)

        memory[address] = value
        self.assertEqual(memory[address], value)

    def test_boundaries(self):
        # exceeding the value and address boundaries are fine
        # as both are masked.
        memory = pixiedust.SQLiteMemory()

        address_boundaries = 0, 0x7fffffff
        value_boundaries = -0x80000000, 0x7fffffff

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

    def test_label_errors(self):
        interpreter = pixiedust.PixieDust()

        bad_label = "".join([random.choice("*+.") for _ in range(10)])
        with self.assertRaisesRegex(SyntaxError, "Invalid label target on line 1"):
            interpreter.execute(
                # jump to bad label
                f"+* + {bad_label}\n"
            )

        good_label = "".join([random.choice("*+.") for _ in range(10)])
        with self.assertRaisesRegex(SyntaxError, "Invalid label target on line 3"):
            interpreter.execute(
                # set good_label
                f"+. {good_label}\n"
                # jnz to good label
                f"+* * {good_label}\n"
                # jump to bad label
                f"+* + {bad_label}\n"
            )


if __name__ == "__main__":
    unittest.main()
