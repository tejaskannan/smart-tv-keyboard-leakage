import unittest
import os.path

from smarttvleakage.audio.tv_classifier import SmartTVTypeClassifier
from smarttvleakage.audio.utils import create_spectrogram
from smarttvleakage.utils.constants import SmartTVType
from smarttvleakage.utils.file_utils import read_pickle_gz


class TVClassificationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.classifier = SmartTVTypeClassifier()

    def test_samsung_test(self):
        self.run_test(path=os.path.join('recordings', 'samsung', 'test.pkl.gz'),
                      expected=SmartTVType.SAMSUNG)
    
    def test_samsung_password(self):
        self.run_test(path=os.path.join('recordings', 'samsung', 'password.pkl.gz'),
                      expected=SmartTVType.SAMSUNG)

    def test_samsung_hello900(self):
        self.run_test(path=os.path.join('recordings', 'samsung', 'hello900.pkl.gz'),
                      expected=SmartTVType.SAMSUNG)

    def test_appletv_qwerty(self):
        self.run_test(path=os.path.join('recordings', 'appletv', 'qwerty.pkl.gz'),
                      expected=SmartTVType.APPLE_TV)

    def test_appletv_lakers(self):
        self.run_test(path=os.path.join('recordings', 'appletv', 'lakers.pkl.gz'),
                      expected=SmartTVType.APPLE_TV)

    def run_test(self, path: str, expected: SmartTVType):
        audio = read_pickle_gz(path)[:, 0]
        target_spectrogram = create_spectrogram(audio)

        result = self.classifier.get_tv_type(target_spectrogram=target_spectrogram)
        self.assertEquals(result, expected)

if __name__ == '__main__':
    unittest.main()
