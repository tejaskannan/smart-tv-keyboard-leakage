import unittest

from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, SAMSUNG_STANDARD, SAMSUNG_SPECIAL_ONE
from smarttvleakage.utils.constants import KeyboardType, Direction


samsung_keyboard = MultiKeyboardGraph(keyboard_type=KeyboardType.SAMSUNG)


class FollowPathTests(unittest.TestCase):

    def test_change_right(self):
        neighbor = samsung_keyboard.follow_path(start_key='<CHANGE>',
                                                use_shortcuts=True,
                                                use_wraparound=True,
                                                directions=[Direction.RIGHT] * 3,
                                                mode=SAMSUNG_STANDARD)
        self.assertEqual(neighbor, 'e')



if __name__ == '__main__':
    unittest.main()
