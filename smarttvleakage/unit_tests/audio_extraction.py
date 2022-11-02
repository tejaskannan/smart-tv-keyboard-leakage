import unittest
import time
import smarttvleakage.audio.sounds as sounds
from smarttvleakage.audio import make_move_extractor
from smarttvleakage.utils.constants import SmartTVType
from smarttvleakage.utils.file_utils import read_pickle_gz


class SamsungAudioExtraction(unittest.TestCase):

    def test_bed(self):
        audio_signal = read_pickle_gz('sounds/samsung/bed.pkl.gz')[:, 0]

        extractor = make_move_extractor(tv_type=SmartTVType.SAMSUNG)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal, include_moves_to_done=False)

        move_seq = list(map(lambda t: t.num_moves, moves))
        self.assertEqual(move_seq, [6, 4, 1])

    def test_elk(self):
        audio_signal = read_pickle_gz('sounds/samsung/elk.pkl.gz')[:, 0]

        extractor = make_move_extractor(tv_type=SmartTVType.SAMSUNG)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal, include_moves_to_done=False)

        move_seq = list(map(lambda t: t.num_moves, moves))
        self.assertEqual(move_seq, [2, 7, 1])

    def test_dog(self):
        audio_signal = read_pickle_gz('sounds/samsung/dog.pkl.gz')[:, 0]

        extractor = make_move_extractor(tv_type=SmartTVType.SAMSUNG)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal, include_moves_to_done=False)

        move_seq = list(map(lambda t: t.num_moves, moves))
        self.assertEqual(move_seq, [3, 7, 5])

    def test_good(self):
        audio_signal = read_pickle_gz('sounds/samsung/good.pkl.gz')[:, 0]

        extractor = make_move_extractor(tv_type=SmartTVType.SAMSUNG)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal, include_moves_to_done=False)

        move_seq = list(map(lambda t: t.num_moves, moves))
        self.assertEqual(move_seq, [5, 5, 0, 7])

    def test_tree(self):
        audio_signal = read_pickle_gz('sounds/samsung/tree.pkl.gz')[:, 0]

        extractor = make_move_extractor(tv_type=SmartTVType.SAMSUNG)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal, include_moves_to_done=False)

        move_seq = list(map(lambda t: t.num_moves, moves))
        self.assertEqual(move_seq, [4, 1, 1, 0])

    def test_soccer3(self):
        audio_signal = read_pickle_gz('sounds/samsung/soccer3.pkl.gz')[:, 0]

        extractor = make_move_extractor(tv_type=SmartTVType.SAMSUNG)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal, include_moves_to_done=False)

        move_seq = list(map(lambda t: t.num_moves, moves))
        self.assertEqual(move_seq, [2, 8, 8, 0, 2, 1, 2])

    def test_full_interaction(self):
        audio_signal = read_pickle_gz('sounds/samsung/full-interaction.pkl.gz')[:, 0]

        extractor = make_move_extractor(tv_type=SmartTVType.SAMSUNG)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal, include_moves_to_done=False)

        move_seq = list(map(lambda t: t.num_moves, moves))
        self.assertEqual(move_seq, [4, 2, 2, 4])

    def test_earth_autocomplete(self):
        audio_signal = read_pickle_gz('sounds/samsung/earth.pkl.gz')[:, 0]

        extractor = make_move_extractor(tv_type=SmartTVType.SAMSUNG)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal, include_moves_to_done=False)

        move_seq = list(map(lambda t: t.num_moves, moves))
        self.assertEqual(move_seq, [2, 4, 1, 0, 0])

    def test_add_autocomplete(self):
        audio_signal = read_pickle_gz('sounds/samsung/add.pkl.gz')[:, 0]

        extractor = make_move_extractor(tv_type=SmartTVType.SAMSUNG)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal, include_moves_to_done=False)

        move_seq = list(map(lambda t: t.num_moves, moves))
        self.assertEqual(move_seq, [1, 3, 0])

    def test_be_autocomplete(self):
        audio_signal = read_pickle_gz('sounds/samsung/be.pkl.gz')[:, 0]

        extractor = make_move_extractor(tv_type=SmartTVType.SAMSUNG)
        moves, did_use_autocomplete, _ = extractor.extract_move_sequence(audio=audio_signal, include_moves_to_done=False)

        move_seq = list(map(lambda t: t.num_moves, moves))
        self.assertEqual(move_seq, [6, 1])
        self.assertTrue(not did_use_autocomplete)

    def test_remember_autocomplete(self):
        audio_signal = read_pickle_gz('sounds/samsung/remember.pkl.gz')[:, 0]

        extractor = make_move_extractor(tv_type=SmartTVType.SAMSUNG)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal, include_moves_to_done=False)

        move_seq = list(map(lambda t: t.num_moves, moves))
        self.assertEqual(move_seq, [3, 1, 1, 0, 0, 0, 0, 0])

    def test_note_autocomplete(self):
        audio_signal = read_pickle_gz('sounds/samsung/note.pkl.gz')[:, 0]

        extractor = make_move_extractor(tv_type=SmartTVType.SAMSUNG)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal, include_moves_to_done=False)

        move_seq = list(map(lambda t: t.num_moves, moves))
        self.assertEqual(move_seq, [7, 1, 1, 1])

    def test_now_autocomplete(self):
        audio_signal = read_pickle_gz('sounds/samsung/now.pkl.gz')[:, 0]

        extractor = make_move_extractor(tv_type=SmartTVType.SAMSUNG)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal, include_moves_to_done=False)

        move_seq = list(map(lambda t: t.num_moves, moves))
        self.assertEqual(move_seq, [7, 1, 0])

    def test_measure_autocomplete(self):
        audio_signal = read_pickle_gz('sounds/samsung/measure.pkl.gz')[:, 0]

        extractor = make_move_extractor(tv_type=SmartTVType.SAMSUNG)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal, include_moves_to_done=False)

        move_seq = list(map(lambda t: t.num_moves, moves))
        self.assertEqual(move_seq, [8, 1, 0, 6, 1, 0, 1])

    def test_year_autocomplete(self):
        audio_signal = read_pickle_gz('sounds/samsung/year.pkl.gz')[:, 0]

        extractor = make_move_extractor(tv_type=SmartTVType.SAMSUNG)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal, include_moves_to_done=False)

        move_seq = list(map(lambda t: t.num_moves, moves))
        self.assertEqual(move_seq, [5, 1, 0, 1])

    def test_warr_backspace(self):
        audio_signal = read_pickle_gz('sounds/samsung/warr.pkl.gz')[:, 0]

        extractor = make_move_extractor(tv_type=SmartTVType.SAMSUNG)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal, include_moves_to_done=False)

        move_seq = list(map(lambda t: t.num_moves, moves))
        self.assertEqual(move_seq, [1, 2, 4, 0, 0, 8])

        sound_seq = list(map(lambda t: t.end_sound, moves))
        expected = [sounds.SAMSUNG_KEY_SELECT, sounds.SAMSUNG_KEY_SELECT, sounds.SAMSUNG_KEY_SELECT, sounds.SAMSUNG_KEY_SELECT, sounds.SAMSUNG_KEY_SELECT, sounds.SAMSUNG_DELETE]
        self.assertEqual(sound_seq, expected)

    #def test_vol_50(self):
    #    audio_signal = read_pickle_gz('sounds/half_volume.pkl.gz')

    #    extractor = make_move_extractor(tv_type=SmartTVType.SAMSUNG)
    #    move_seq, _ = extractor.extract_move_sequence(audio=audio_signal)

    #    self.assertEqual(move_seq, [4, 2, 2, 4])


class AppleTVAudioExtraction(unittest.TestCase):

    def test_roar(self):
        audio_signal = read_pickle_gz('sounds/apple_tv/roar.pkl.gz')

        extractor = make_move_extractor(tv_type=SmartTVType.APPLE_TV)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal, include_moves_to_done=False)

        move_seq = list(map(lambda t: t.num_moves, moves))
        self.assertEqual(move_seq, [3, 3, 14, 17])

    def test_wecrashed(self):
        audio_signal = read_pickle_gz('sounds/apple_tv/wecrashed.pkl.gz')

        extractor = make_move_extractor(tv_type=SmartTVType.APPLE_TV)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal, include_moves_to_done=False)

        move_seq = list(map(lambda t: t.num_moves, moves))
        self.assertEqual(move_seq, [4, 18, 2, 15, 17, 18, 11, 3, 1])

    def test_hello_scroll(self):
        audio_signal = read_pickle_gz('sounds/apple_tv/hello_scroll.pkl.gz')

        extractor = make_move_extractor(tv_type=SmartTVType.APPLE_TV)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal, include_moves_to_done=False)

        move_seq = list(map(lambda t: t.num_moves, moves))
        self.assertEqual(move_seq, [13, 3, 7, 0, 5])

    def test_test_scroll(self):
        audio_signal = read_pickle_gz('sounds/apple_tv/test_scroll.pkl.gz')

        extractor = make_move_extractor(tv_type=SmartTVType.APPLE_TV)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal, include_moves_to_done=False)

        move_seq = list(map(lambda t: t.num_moves, moves))
        self.assertEqual(move_seq, [1, 15, 16, 1])

    def test_star_trek_backspace(self):
        audio_signal = read_pickle_gz('sounds/apple_tv/star_trek.pkl.gz')

        extractor = make_move_extractor(tv_type=SmartTVType.APPLE_TV)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal, include_moves_to_done=False)

        move_seq = list(map(lambda t: t.num_moves, moves))
        self.assertEqual(move_seq, [2, 1, 19, 17, 0, 9, 27, 20, 2, 13, 6])

        sound_seq = list(map(lambda t: t.end_sound, moves))
        expected = [sounds.APPLETV_KEYBOARD_SELECT, sounds.APPLETV_KEYBOARD_SELECT, sounds.APPLETV_KEYBOARD_SELECT, sounds.APPLETV_KEYBOARD_SELECT, sounds.APPLETV_KEYBOARD_SELECT, sounds.APPLETV_KEYBOARD_DELETE, sounds.APPLETV_KEYBOARD_SELECT, sounds.APPLETV_KEYBOARD_SELECT, sounds.APPLETV_KEYBOARD_SELECT, sounds.APPLETV_KEYBOARD_SELECT, sounds.APPLETV_KEYBOARD_SELECT]
        self.assertEqual(sound_seq, expected)


if __name__ == '__main__':
    unittest.main()
