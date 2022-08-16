import unittest
from smarttvleakage.utils.edit_distance import compute_edit_distance


class EditDistanceTests(unittest.TestCase):

    def test_empty(self):
        distance = compute_edit_distance('', '')
        self.assertEqual(distance, 0)

        distance = compute_edit_distance('', 'abcd')
        self.assertEqual(distance, 4)

        distance = compute_edit_distance('abc', '')
        self.assertEqual(distance, 3)

    def test_equal(self):
        distance = compute_edit_distance('abc', 'abc')
        self.assertEqual(distance, 0)

    def test_single(self):
        distance = compute_edit_distance('abcd', 'abdd')
        self.assertEqual(distance, 1)

        distance = compute_edit_distance('abcd', 'abd')
        self.assertEqual(distance, 1)

        distance = compute_edit_distance('abcd', 'abbcd')
        self.assertEqual(distance, 1)

    def test_double(self):
        distance = compute_edit_distance('their', 'there')
        self.assertEqual(distance, 2)

    def test_triple(self):
        distance = compute_edit_distance('kitten', 'sitting')
        self.assertEqual(distance, 3)


if __name__ == '__main__':
    unittest.main()

