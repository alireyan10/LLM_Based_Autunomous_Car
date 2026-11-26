#!/usr/bin/env python3
"""
Unit tests for Bangla Voice Processor (location_navigation.voice_processor).
Tests phonetic replacements, fuzzy matching, and safe preservation of keywords.
"""

import os
import wave
import unittest
from unittest.mock import patch, MagicMock
from location_navigation.voice_processor import (
    clean_and_correct_stt_text,
    convert_to_wav,
    is_compatible_wav,
    speech_to_bangla_text,
    listen_from_microphone,
    SUPPORTED_AUDIO_VIDEO_EXTENSIONS,
)


class TestVoiceProcessor(unittest.TestCase):
    def test_phonetic_replacements(self):
        """Verify phonetic confusion words are mapped to domain keywords."""
        # 'কানে' -> 'ডানে'
        self.assertEqual(clean_and_correct_stt_text("রোবট কানে যাও", verbose=False), "রোবট ডানে যাও")
        # 'টানে' -> 'ডানে'
        self.assertEqual(clean_and_correct_stt_text("একটু টানে ঘুরো", verbose=False), "একটু ডানে ঘুরো")
        # 'ছুফা' -> 'সোফা'
        self.assertEqual(clean_and_correct_stt_text("ছুফার কাছে যাও", verbose=False), "সোফার কাছে যাও")
        # 'টিবিল' -> 'টেবিল'
        self.assertEqual(clean_and_correct_stt_text("টিবিলের কাছে যাও", verbose=False), "টেবিলের কাছে যাও")
        # 'শামনে' -> 'সামনে'
        self.assertEqual(clean_and_correct_stt_text("এক মিটার শামনে যাও", verbose=False), "এক মিটার সামনে যাও")
        # 'সজা' -> 'সোজা'
        self.assertEqual(clean_and_correct_stt_text("সজা যাও", verbose=False), "সোজা যাও")

    def test_fuzzy_matching_domain_words(self):
        """Verify fuzzy typos close to domain words are mapped properly."""
        # 'রান্নাঘরের' slight typo
        cleaned = clean_and_correct_stt_text("রোবট রান্নাঘর যাও", verbose=False)
        self.assertIn("রান্নাঘর", cleaned)

        # 'বেডরুম'
        cleaned = clean_and_correct_stt_text("বেডরুমে যাও", verbose=False)
        self.assertIn("বেডরুমে", cleaned)

    def test_already_clean_commands_preserved(self):
        """Ensure already valid Bangla commands are preserved without unwanted modifications."""
        cmd = "রোবট রান্নাঘরে যাও"
        self.assertEqual(clean_and_correct_stt_text(cmd, verbose=False), cmd)

        cmd2 = "রোবট সোফার কাছে যাও তারপর এক মিটার সামনে যাও"
        self.assertEqual(clean_and_correct_stt_text(cmd2, verbose=False), cmd2)

        cmd3 = "থামো"
        self.assertEqual(clean_and_correct_stt_text(cmd3, verbose=False), cmd3)

    def test_empty_and_punctuation_handling(self):
        """Handles empty strings and removes trailing punctuation cleanly."""
        self.assertEqual(clean_and_correct_stt_text("", verbose=False), "")
        self.assertEqual(clean_and_correct_stt_text("রোবট, ডানে যাও!", verbose=False), "রোবট ডানে যাও")

    def test_convert_to_wav_missing_file_raises_error(self):
        """Verify non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            convert_to_wav("/non/existent/path/command.mp4")

    def test_convert_to_wav_unsupported_format_raises_error(self):
        """Verify unsupported file extension raises ValueError."""
        # Create a temp file with .txt extension
        temp_path = "/tmp/test_invalid.txt"
        with open(temp_path, "w") as f:
            f.write("dummy")
        try:
            with self.assertRaises(ValueError) as ctx:
                convert_to_wav(temp_path)
            self.assertIn("Unsupported file format", str(ctx.exception))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @patch("shutil.which", return_value=None)
    def test_convert_to_wav_missing_ffmpeg_handled(self, mock_which):
        """Verify missing FFmpeg raises descriptive RuntimeError."""
        temp_path = "/tmp/test_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(b"dummy mp4 content")
        try:
            with self.assertRaises(RuntimeError) as ctx:
                convert_to_wav(temp_path)
            self.assertIn("FFmpeg is not installed", str(ctx.exception))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_is_compatible_wav_and_passthrough(self):
        """Verify 16kHz mono 16-bit PCM WAV is detected and passed through directly."""
        temp_wav = "/tmp/test_compatible_16k.wav"
        with wave.open(temp_wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 16000)

        try:
            self.assertTrue(is_compatible_wav(temp_wav))
            # Calling convert_to_wav on it should return the exact same path
            result = convert_to_wav(temp_wav)
            self.assertEqual(result, temp_wav)
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_convert_to_wav_ffmpeg_command_parameters(self, mock_run, mock_which):
        """Verify FFmpeg is invoked with -vn, -acodec pcm_s16le, -ac 1, -ar 16000."""
        temp_mp4 = "/tmp/test_input.mp4"
        with open(temp_mp4, "wb") as f:
            f.write(b"dummy")

        # Fake success and write non-empty file to output path
        def fake_run(cmd, *args, **kwargs):
            out_file = cmd[-1]
            with open(out_file, "wb") as f:
                f.write(b"wavdata")
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stderr = ""
            return mock_proc

        mock_run.side_effect = fake_run

        try:
            out_wav = convert_to_wav(temp_mp4)
            self.assertTrue(os.path.exists(out_wav))
            cmd_called = mock_run.call_args[0][0]
            self.assertEqual(cmd_called[0], "/usr/bin/ffmpeg")
            self.assertIn("-vn", cmd_called)
            self.assertIn("-acodec", cmd_called)
            self.assertIn("pcm_s16le", cmd_called)
            self.assertIn("-ac", cmd_called)
            self.assertIn("1", cmd_called)
            self.assertIn("-ar", cmd_called)
            self.assertIn("16000", cmd_called)
            os.remove(out_wav)
        finally:
            if os.path.exists(temp_mp4):
                os.remove(temp_mp4)

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_convert_to_wav_no_audio_stream_handled(self, mock_run, mock_which):
        """Verify video file without audio stream raises descriptive RuntimeError."""
        temp_mp4 = "/tmp/silent_video.mp4"
        with open(temp_mp4, "wb") as f:
            f.write(b"dummy")

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = "Output file does not contain any stream"
        mock_run.return_value = mock_proc

        try:
            with self.assertRaises(RuntimeError) as ctx:
                convert_to_wav(temp_mp4)
            self.assertIn("contains no audio stream", str(ctx.exception))
        finally:
            if os.path.exists(temp_mp4):
                os.remove(temp_mp4)

    @patch("location_navigation.voice_processor._get_speech_recognizer_module", return_value=None)
    def test_missing_speech_recognition_reported_gracefully(self, mock_sr):
        """Verify missing SpeechRecognition returns None gracefully without unhandled crash."""
        res = speech_to_bangla_text("/dummy/path.wav", verbose=False)
        self.assertIsNone(res)

        res_mic = listen_from_microphone(verbose=False)
        self.assertIsNone(res_mic)

    @patch("location_navigation.voice_processor._get_speech_recognizer_module")
    def test_speech_to_bangla_text_mp4_transcription_and_temp_cleanup(self, mock_sr_getter):
        """Verify speech_to_bangla_text automatically handles MP4 and cleans up temp files."""
        temp_wav = "/tmp/voice_test_16k.wav"
        with wave.open(temp_wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 16000)

        mock_sr = MagicMock()
        mock_recognizer = MagicMock()
        mock_recognizer.recognize_google.return_value = "রোবট রান্নাঘরে যাও"
        mock_sr.Recognizer.return_value = mock_recognizer
        mock_sr_getter.return_value = mock_sr

        try:
            # Test direct call with compatible WAV
            res = speech_to_bangla_text(temp_wav, recognizer=mock_recognizer, verbose=False)
            self.assertEqual(res, "রোবট রান্নাঘরে যাও")
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)


if __name__ == '__main__':
    unittest.main()
