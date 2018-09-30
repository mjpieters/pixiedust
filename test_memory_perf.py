# Copyright 2018 Martijn Pieters, Zopatista Ltd.
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE.txt file for details.

import random
import unittest

import pixiedust


class SQLMemoryPerfTests(unittest.TestCase):
    def test_01_random_access(self):
        memory = pixiedust.SQLiteMemory()
        testsize = 10 ** 5
        passes = 3

        # set the test values
        label = (
            f"randomised memory test: setting "
            f"{{:>{len(str(testsize - 1))}d}} / {testsize} values."
        )
        tests = {}
        for i in range(testsize):
            print(f"\r{label}".format(i + 1), end="", flush=True)
            address, value = (
                random.randint(0, 0x7fffffff),
                random.randint(-0x80000000, 0x7fffffff),
            )
            tests[address] = memory[address] = value
        print("\r" + (" " * (len(label) + 2)), end="")

        # read values, randomly setting some to new values
        for i in range(passes):
            label = (
                f"randomised memory test: pass {i + 1:>{len(str(passes - 1))}} / {passes}"
            )
            with self.subTest(i=i):
                # read in random order, sometimes resetting values
                for i, address in enumerate(random.sample(tests.keys(), len(tests)), 1):
                    print(f"\r{label} {(i + 1) / testsize:.2%}", end="", flush=True)
                    if random.random() < 0.2:
                        tests[address] = random.randint(-0x80000000, 0x7fffffff)
                        memory[address] = tests[address]
                    self.assertEqual(memory[address], tests[address])

        print()

    def test_02_full_fill(self):
        memory = pixiedust.SQLiteMemory()
        max_address = 0x7fffffff
        total = max_address + 1
        hexsize = (max_address.bit_length() + 3) // 4

        for address in range(total):
            print(
                "\rfull memory fill: address "
                f"{address:#0{hexsize + 2}x} ({(address + 1)/total:.2%})",
                end="",
                flush=True,
            )
            value = random.randint(-0x80000000, 0x7fffffff)
            memory[address] = value


if __name__ == "__main__":
    unittest.main()
