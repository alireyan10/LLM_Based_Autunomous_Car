#!/usr/bin/env python3
"""
Bangla Voice Processing Module for location_navigation.

Transcribes Bangla speech to text using Google Speech Recognition (bn-BD)
and applies domain-aware phonetic and fuzzy noise auto-correction
to align transcription with the robot's vocabulary (targets, actions, directions).
"""

import os
import sys
import difflib
import subprocess
import shutil
import tempfile
import wave
from typing import Optional, List, Tuple, Set

# Domain vocabulary for mobile robot navigation
DOMAIN_DIRECTIONS = [
    "ডানে", "ডানদিকে", "ডান", "বামে", "বামদিকে", "বাম",
    "সামনে", "সামনের", "সোজা", "পেছনে", "পিছনে", "পিছনের"
]

DOMAIN_ACTIONS = [
    "যাও", "ঘুরো", "থামো", "দাঁড়াও", "চলো", "আস্তে", "ধীরে", "দ্রুত", "গতি", "যাবে", "চল"
]

DOMAIN_TARGETS = [
    "টেবিল", "টেবিলে", "টেবিলের", "ডাইনিং টেবিল", "ডাইনিং", "খাবার টেবিল",
    "সোফা", "সোফায়", "সোফার", "বিছানা", "বিছানায়", "বিছানার", "খাট", "খাটে", "খাটের",
    "দরজা", "দরজার", "চেয়ার", "চেয়ারের", "টিভি", "টেলিভিশন", "ফ্রিজ", "ফ্রিজে", "ফ্রিজের",
    "রেফ্রিজারেটর", "রান্নাঘর", "রান্নাঘরে", "বেডরুম", "বেডরুমে", "শোবার ঘর", "শোবার ঘরে"
]

DOMAIN_UNITS = [
    "মিটার", "মিটারে", "ইঞ্চি", "কদম", "ডিগ্রি", "ডিগ্রী",
    "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়", "দশ",
    "১", "২", "৩", "৪", "৫", "৬", "৭", "৮", "৯", "১০",
    "০.৫", "১.৫", "২.৫", "০.২", "০.৩"
]

COMMON_SAFE_WORDS = {
    "রোবট", "তুমি", "তারপর", "পরে", "আর", "একটু", "কাছে", "দিকে",
    "থেকে", "গিয়ে", "করে", "নিয়ে", "এবং", "এখন", "প্লিজ"
}

ALL_DOMAIN_KEYWORDS = DOMAIN_DIRECTIONS + DOMAIN_ACTIONS + DOMAIN_TARGETS + DOMAIN_UNITS

# Common phonetic STT confusions for Bangla speech recognition
PHONETIC_REPLACEMENTS = {
    "কানে": "ডানে",
    "টানে": "ডানে",
    "দানে": "ডানে",
    "বানে": "বামে",
    "বায়ে": "বামে",
    "শামনে": "সামনে",
    "সজা": "সোজা",
    "ছুফা": "সোফা",
    "টিবিল": "টেবিল",
    "দাড়াও": "দাঁড়াও",
    "রানাঘর": "রান্নাঘর",
    "ফিজ": "ফ্রিজ",
    "বেডরম": "বেডরুম",
    "থাটো": "থামো",
}

BANGLA_INFLECTION_SUFFIXES = ["ের", "র", "য়", "য়ে", "তে", "টি", "টা"]


def clean_and_correct_stt_text(raw_text: str, verbose: bool = True) -> str:
    """
    Cleans raw STT text by mapping phonetic confusions and fuzzy typos
    to domain-relevant robotics vocabulary. Handles inflected words (e.g. ছুফার -> সোফার).
    """
    if not raw_text or not isinstance(raw_text, str):
        return ""

    words = raw_text.split()
    corrected_words = []
    corrections_made = []

    for word in words:
        clean_word = word.strip("।,?!:;\"' ")

        # 1. Direct phonetic STT error dictionary
        if clean_word in PHONETIC_REPLACEMENTS:
            fixed = PHONETIC_REPLACEMENTS[clean_word]
            corrected_words.append(fixed)
            corrections_made.append(f"'{clean_word}' -> '{fixed}' (phonetic)")
            continue

        # 1b. Stem-level phonetic check for inflected words (e.g. ছুফার -> সোফার, টিবিলের -> টেবিলের)
        stem_fixed = None
        for suffix in BANGLA_INFLECTION_SUFFIXES:
            if clean_word.endswith(suffix) and len(clean_word) > len(suffix):
                stem = clean_word[:-len(suffix)]
                if stem in PHONETIC_REPLACEMENTS:
                    stem_fixed = PHONETIC_REPLACEMENTS[stem] + suffix
                    break
        if stem_fixed is not None:
            corrected_words.append(stem_fixed)
            corrections_made.append(f"'{clean_word}' -> '{stem_fixed}' (phonetic stem)")
            continue

        # 2. Known domain word or safe common particle
        if clean_word in ALL_DOMAIN_KEYWORDS or clean_word in COMMON_SAFE_WORDS:
            corrected_words.append(clean_word)
            continue

        # 3. Fuzzy matching with domain lexicon (edit distance threshold 0.70)
        fuzzy_matches = difflib.get_close_matches(clean_word, ALL_DOMAIN_KEYWORDS, n=1, cutoff=0.70)
        if fuzzy_matches:
            fixed = fuzzy_matches[0]
            corrected_words.append(fixed)
            corrections_made.append(f"'{clean_word}' -> '{fixed}' (fuzzy)")
        else:
            corrected_words.append(clean_word)

    final_text = " ".join(corrected_words)
    if verbose and corrections_made:
        print(f"[STT Cleaner] Corrected: {', '.join(corrections_made)}")
        print(f"[STT Cleaner] Cleaned Text: \"{final_text}\"")

    return final_text


SUPPORTED_AUDIO_VIDEO_EXTENSIONS: Set[str] = {
    ".wav", ".mp4", ".mp3", ".m4a", ".aac", ".flac",
    ".ogg", ".wma", ".mkv", ".avi", ".mov", ".webm"
}


def _get_speech_recognizer_module():
    """
    Safely imports speech_recognition.
    If not available in standard sys.path, checks active virtualenv or
    the case_study/voice_env virtual environment.
    """
    try:
        import speech_recognition as sr
        return sr
    except ImportError:
        pass

    # Fallback: attempt to locate voice_env site-packages
    candidate_paths = []
    if "VIRTUAL_ENV" in os.environ:
        venv = os.environ["VIRTUAL_ENV"]
        for py_ver in [f"python{sys.version_info.major}.{sys.version_info.minor}", "python3.12", "python3"]:
            sp = os.path.join(venv, "lib", py_ver, "site-packages")
            if os.path.isdir(sp) and sp not in candidate_paths:
                candidate_paths.append(sp)

    # Standard case_study workspace voice_env path
    home = os.path.expanduser("~")
    candidate_paths.append(os.path.join(home, "case_study", "voice_env", "lib", "python3.12", "site-packages"))

    # Workspace-relative search (up to 5 directory levels)
    curr = os.path.abspath(__file__)
    for _ in range(5):
        curr = os.path.dirname(curr)
        sp = os.path.join(curr, "voice_env", "lib", "python3.12", "site-packages")
        if os.path.isdir(sp) and sp not in candidate_paths:
            candidate_paths.append(sp)

    for sp in candidate_paths:
        if os.path.isdir(sp) and sp not in sys.path:
            sys.path.insert(0, sp)
            try:
                import speech_recognition as sr
                return sr
            except ImportError:
                continue

    return None


def is_compatible_wav(file_path: str) -> bool:
    """
    Checks if a WAV file is already in the required format:
    16,000 Hz sample rate, mono channel (1 channel), and 16-bit PCM encoding (2 bytes/sample).
    """
    if not isinstance(file_path, str) or not file_path.lower().endswith(".wav"):
        return False
    if not os.path.isfile(file_path):
        return False
    try:
        with wave.open(file_path, "rb") as w:
            return (
                w.getnchannels() == 1
                and w.getsampwidth() == 2
                and w.getframerate() == 16000
            )
    except Exception:
        return False


def convert_to_wav(input_path: str, output_path: Optional[str] = None) -> str:
    """
    Converts audio/video input files into 16kHz Mono 16-bit PCM WAV using FFmpeg.

    Requirements fulfilled:
    1. Accepts input file path.
    2. Detects file format.
    3. Extracts audio from video files such as MP4 (-vn).
    4. Converts audio into WAV, 16kHz, mono, 16-bit PCM (-acodec pcm_s16le -ac 1 -ar 16000).
    5. Returns converted WAV path.
    6. Handles missing FFmpeg cleanly.
    7. Handles invalid files and unsupported formats.
    8. Handles files without an audio stream.
    9. Reports conversion failures clearly.
    10. Cleans up temporary files safely if conversion fails.
    The original file remains unchanged.
    """
    if not input_path or not isinstance(input_path, str):
        raise ValueError("A valid input file path must be provided.")

    input_path = os.path.abspath(os.path.expanduser(input_path))

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not os.path.isfile(input_path):
        raise ValueError(f"Path is not a regular file: {input_path}")

    _, ext = os.path.splitext(input_path)
    if ext.lower() not in SUPPORTED_AUDIO_VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext}'. Supported formats: "
            f"{', '.join(sorted(SUPPORTED_AUDIO_VIDEO_EXTENSIONS))}"
        )

    # Return immediately if file is already a compatible WAV and no custom output requested
    if output_path is None and is_compatible_wav(input_path):
        return input_path

    # Verify FFmpeg is installed and accessible in PATH
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError(
            "FFmpeg is not installed or not found in system PATH. "
            "Please install FFmpeg (e.g. 'sudo apt install ffmpeg')."
        )

    # Prepare output path
    created_temp = False
    if output_path is None:
        fd, temp_dest = tempfile.mkstemp(prefix="voice_nav_", suffix=".wav")
        os.close(fd)
        output_path = temp_dest
        created_temp = True
    else:
        output_path = os.path.abspath(os.path.expanduser(output_path))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
        ffmpeg_bin,
        "-y",                   # Overwrite output without prompting
        "-i", input_path,       # Input audio/video file
        "-vn",                  # Ignore/disable video track (extract audio only)
        "-acodec", "pcm_s16le", # 16-bit PCM encoding
        "-ac", "1",             # Mono channel
        "-ar", "16000",         # 16,000 Hz sample rate
        output_path
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except Exception as e:
        if created_temp and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        raise RuntimeError(f"Failed to execute FFmpeg: {e}")

    if result.returncode != 0:
        if created_temp and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        stderr = result.stderr or ""
        stderr_lower = stderr.lower()
        if "does not contain any stream" in stderr_lower or "matches no streams" in stderr_lower:
            raise RuntimeError(f"Input file '{input_path}' contains no audio stream to convert.")
        raise RuntimeError(f"FFmpeg conversion failed (exit code {result.returncode}): {stderr.strip()}")

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        if created_temp and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        raise RuntimeError(f"FFmpeg conversion produced an empty or missing output file: {output_path}")

    return output_path


def speech_to_bangla_text(
    audio_path: str,
    recognizer=None,
    verbose: bool = True
) -> Optional[str]:
    """
    Transcribes audio/video input (WAV, MP4, MP3, etc.) to Bengali text using Google Speech API.
    Automatically detects format, extracts audio with FFmpeg if needed, converts to 16kHz mono WAV,
    and safely cleans up any temporary files upon completion.
    """
    sr = _get_speech_recognizer_module()
    if sr is None:
        print("[Voice Error] 'SpeechRecognition' is not installed.", file=sys.stderr)
        print("Install via: pip install SpeechRecognition", file=sys.stderr)
        return None

    if not audio_path or not isinstance(audio_path, str):
        print(f"[Voice Error] Invalid audio path provided: {audio_path}", file=sys.stderr)
        return None

    audio_path = os.path.abspath(os.path.expanduser(audio_path))
    if not os.path.exists(audio_path):
        print(f"[Voice Error] Audio file not found: {audio_path}", file=sys.stderr)
        return None

    converted_wav = None
    try:
        # Convert audio/video file to 16kHz mono 16-bit PCM WAV if needed
        wav_path = convert_to_wav(audio_path)
        if wav_path != audio_path:
            converted_wav = wav_path
    except Exception as e:
        if verbose:
            print(f"[Voice Error] Failed to prepare audio for transcription: {e}", file=sys.stderr)
        return None

    try:
        if recognizer is None:
            recognizer = sr.Recognizer()

        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)

        if verbose:
            print("[STT] Transcribing Bangla speech with Google API...")
        raw_text = recognizer.recognize_google(audio_data, language="bn-BD")
        if verbose:
            print(f"[STT Raw]: \"{raw_text}\"")
        return clean_and_correct_stt_text(raw_text, verbose=verbose)
    except sr.UnknownValueError:
        if verbose:
            print("[STT] Audio was unclear or unintelligible.", file=sys.stderr)
        return None
    except sr.RequestError as e:
        if verbose:
            print(f"[STT Error] Google Speech API request failed: {e}", file=sys.stderr)
        return None
    except Exception as e:
        if verbose:
            print(f"[STT Error]: {e}", file=sys.stderr)
        return None
    finally:
        # Safely clean up temporary converted audio file
        if converted_wav and os.path.exists(converted_wav):
            try:
                os.remove(converted_wav)
            except OSError:
                pass


def listen_from_microphone(
    timeout: float = 8.0,
    phrase_time_limit: float = 12.0,
    recognizer=None,
    verbose: bool = True
) -> Optional[str]:
    """
    Listens live from microphone and transcribes Bangla speech to cleaned text.
    Preserves existing microphone input functionality while reporting missing dependencies clearly.
    """
    sr = _get_speech_recognizer_module()
    if sr is None:
        print("[Voice Error] 'SpeechRecognition' is not installed.", file=sys.stderr)
        print("Install via: pip install SpeechRecognition", file=sys.stderr)
        return None

    # Check PyAudio explicitly
    try:
        import pyaudio
    except ImportError:
        print("[Voice Error] 'PyAudio' is not installed.", file=sys.stderr)
        print("Install via: sudo apt install python3-pyaudio or pip install pyaudio", file=sys.stderr)
        return None

    if recognizer is None:
        recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            if verbose:
                print("\n" + "=" * 60)
                print("🎤 MICROPHONE ACTIVE: Speak your Bangla command...")
                print("   (Example: 'রোবট রান্নাঘরে যাও' or 'টেবিলের কাছে যাও')")
                print("=" * 60)
            recognizer.adjust_for_ambient_noise(source, duration=0.6)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    except sr.WaitTimeoutError:
        if verbose:
            print("[Mic] No speech detected within timeout period.")
        return None
    except Exception as e:
        if verbose:
            print(f"[Mic Error] Could not access microphone: {e}", file=sys.stderr)
            print("Ensure your microphone is plugged in and PyAudio is working properly.")
        return None

    try:
        if verbose:
            print("[STT] Recognizing speech...")
        raw_text = recognizer.recognize_google(audio, language="bn-BD")
        if verbose:
            print(f"[STT Raw]: \"{raw_text}\"")
        clean_text = clean_and_correct_stt_text(raw_text, verbose=verbose)
        if verbose:
            print(f"[STT Final Cleaned]: \"{clean_text}\"")
        return clean_text
    except sr.UnknownValueError:
        if verbose:
            print("[STT] Speech unclear or no sound detected.")
        return None
    except Exception as e:
        if verbose:
            print(f"[STT Error]: {e}", file=sys.stderr)
        return None
