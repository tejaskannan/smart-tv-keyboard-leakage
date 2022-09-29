from smarttvleakage.utils.constants import KeyboardType

# Samsung Smart TV Sounds
SAMSUNG_DELETE = 'delete'
SAMSUNG_DOUBLE_MOVE = 'double_move'
SAMSUNG_KEY_SELECT = 'key_select'
SAMSUNG_MOVE = 'move'
SAMSUNG_SELECT = 'select'


# Apple TV Sounds
APPLETV_KEYBOARD_DELETE = 'keyboard_delete'
APPLETV_KEYBOARD_MOVE = 'keyboard_move'
APPLETV_KEYBOARD_DOUBLE_MOVE = 'keyboard_double_move'
APPLETV_KEYBOARD_SCROLL_DOUBLE = 'keyboard_scroll_double'
APPLETV_KEYBOARD_SCROLL_TRIPLE = 'keyboard_scroll_triple'
APPLETV_KEYBOARD_SELECT = 'keyboard_select'
APPLETV_SYSTEM_MOVE = 'system_move'
APPLETV_TOOLBAR_MOVE = 'toolbar_move'


# Collect all sounds for each system
SAMSUNG_SOUNDS = frozenset([SAMSUNG_DOUBLE_MOVE, SAMSUNG_KEY_SELECT, SAMSUNG_MOVE, SAMSUNG_SELECT, SAMSUNG_DELETE])
APPLETV_SOUNDS = frozenset([APPLETV_KEYBOARD_MOVE, APPLETV_KEYBOARD_DOUBLE_MOVE, APPLETV_KEYBOARD_SELECT, APPLETV_SYSTEM_MOVE, APPLETV_KEYBOARD_DELETE, APPLETV_TOOLBAR_MOVE, APPLETV_KEYBOARD_SCROLL_DOUBLE, APPLETV_KEYBOARD_SCROLL_TRIPLE])


# Make a dictionary for the sounds made on certain `special` keys
DONE_SOUNDS = {
    KeyboardType.SAMSUNG: SAMSUNG_SELECT,
    KeyboardType.APPLE_TV_PASSWORD: APPLETV_TOOLBAR_MOVE,
    KeyboardType.APPLE_TV_SEARCH: APPLETV_SYSTEM_MOVE
}


SPACE_SOUNDS = {
    KeyboardType.SAMSUNG: SAMSUNG_SELECT,
    KeyboardType.APPLE_TV_PASSWORD: APPLETV_KEYBOARD_SELECT,
    KeyboardType.APPLE_TV_SEARCH: APPLETV_KEYBOARD_SELECT
}


CHANGE_SOUNDS = {
    KeyboardType.SAMSUNG: SAMSUNG_SELECT,
    KeyboardType.APPLE_TV_PASSWORD: APPLETV_KEYBOARD_SELECT,
    KeyboardType.APPLE_TV_SEARCH: APPLETV_KEYBOARD_SELECT
}


KEY_SELECT_SOUNDS = {
    KeyboardType.SAMSUNG: SAMSUNG_KEY_SELECT,
    KeyboardType.APPLE_TV_PASSWORD: APPLETV_KEYBOARD_SELECT,
    KeyboardType.APPLE_TV_SEARCH: APPLETV_KEYBOARD_SELECT
}
