import unittest
from src.app import main

class TestApp(unittest.TestCase):
    def test_main(self):
        with self.assertLogs() as cm:
            main()
        self.assertEqual(cm.output, ['INFO:root:Hello, world!'])

if __name__ == '__main__':
    unittest.main()
