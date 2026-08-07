import unittest

import numpy as np

from hausa_s2tt.audio import (
    AudioValidationError,
    iter_audio_chunks,
    resample_audio,
    to_mono,
    validate_audio,
)


class AudioTests(unittest.TestCase):
    def test_stereo_frames_channels_to_mono(self):
        stereo = np.array([[1.0, -1.0], [0.5, 0.5]], dtype=np.float32)
        np.testing.assert_allclose(to_mono(stereo), np.array([0.0, 0.5]))

    def test_channels_frames_to_mono(self):
        stereo = np.vstack([np.ones(100), np.zeros(100)]).astype(np.float32)
        mono = to_mono(stereo)
        self.assertEqual(mono.shape, (100,))
        np.testing.assert_allclose(mono, 0.5)

    def test_resampling_has_expected_length(self):
        source = np.linspace(-1, 1, 8000, dtype=np.float32)
        target = resample_audio(source, 8000, 16000)
        self.assertEqual(len(target), 16000)
        self.assertEqual(target.dtype, np.float32)

    def test_empty_and_nonfinite_audio_rejected(self):
        with self.assertRaises(AudioValidationError):
            validate_audio(np.array([], dtype=np.float32), 16000)
        with self.assertRaises(AudioValidationError):
            validate_audio(np.array([0.0, np.nan], dtype=np.float32), 16000)

    def test_long_audio_is_chunked_not_truncated(self):
        samples = np.zeros(65 * 16000, dtype=np.float32)
        chunks = list(iter_audio_chunks(samples, 16000, chunk_length_seconds=29))
        self.assertEqual(len(chunks), 3)
        self.assertAlmostEqual(chunks[-1][2], 65.0)


if __name__ == "__main__":
    unittest.main()
