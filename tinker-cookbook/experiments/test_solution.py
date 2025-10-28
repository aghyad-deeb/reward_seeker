import unittest
from experiments.solution import sample_function1, sample_function2

class TestSolution(unittest.TestCase):
    def test_sample_function1(self):
        self.assertEqual(sample_function1(5), 25)
        self.assertEqual(sample_function1(-3), 9)

    def test_sample_function2(self):
        self.assertEqual(sample_function2(3, 4), 7)
        self.assertEqual(sample_function2(-1, 5), 4)

if __name__ == '__main__':
    unittest.main()
