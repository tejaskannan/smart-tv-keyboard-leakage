import unittest

from smarttvleakage.utils.transformations import capitalization_combinations


class Transformations(unittest.TestCase):

    def test_capitalizations(self):
        string = '5fGE'
        expected = { '5fGE', '5FGE', '5FgE', '5FGe', '5Fge', '5fGe', '5fgE', '5fge' }
        result = capitalization_combinations(string)

        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()

