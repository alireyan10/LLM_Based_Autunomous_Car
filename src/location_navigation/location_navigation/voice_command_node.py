#!/usr/bin/env python3
"""
Bangla Voice Command Navigation Node for ROS 2 (car_bot).

This node listens to voice commands via microphone (or audio file),
converts the spoken Bangla command to text with automatic noise cleaning,
and forwards it to the Qwen LLM navigation pipeline (llm_command_node -> llm_to_nav -> Nav2).
"""

import os
import sys
import argparse
import rclpy
from rclpy.node import Node

from location_navigation.voice_processor import (
    clean_and_correct_stt_text,
    convert_to_wav,
    speech_to_bangla_text,
    listen_from_microphone,
)
from location_navigation.llm_command_node import process_command


def run_voice_navigation(audio_file: str = None, continuous: bool = False):
    """
    Main loop or single execution for voice-driven navigation.
    """
    if audio_file:
        print(f"\n[Voice Nav] Processing input audio file: {audio_file}")
        temp_wav = None
        try:
            # Automatically detects format and converts to 16kHz mono WAV if needed
            wav_path = convert_to_wav(audio_file)
            input_abs = os.path.abspath(os.path.expanduser(audio_file))
            if wav_path != input_abs:
                temp_wav = wav_path
            cleaned_text = speech_to_bangla_text(wav_path)
            if cleaned_text:
                print(f"[Voice Nav] Recognized command: \"{cleaned_text}\"")
                process_command(cleaned_text)
            else:
                print("[Voice Nav] No clear command transcribed from audio file.")
        except Exception as e:
            print(f"[Voice Nav] Error processing audio file: {e}")
        finally:
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except OSError:
                    pass
        return

    # Live Microphone Mode
    print("\n" + "=" * 65)
    print("🤖 CAR_BOT BANGLA VOICE NAVIGATION SYSTEM")
    print("=" * 65)
    print("Commands supported: Locations (রান্নাঘর, বেডরুম, সোফা, টেবিল),")
    print("                     Motions (১ মিটার সামনে যাও, ৯০ ডিগ্রি ডানে ঘুরো, ইত্যাদি)")
    print("=" * 65)

    while rclpy.ok():
        try:
            if not continuous:
                user_prompt = input("\n[Press Enter to Speak, or type 'exit' to quit]: ").strip()
                if user_prompt.lower() in ("exit", "quit", "q"):
                    break
                if user_prompt and user_prompt.lower() not in ("", "s", "speak"):
                    # Fallback: user typed text instead of speaking
                    process_command(user_prompt)
                    continue

            print("\nListening for voice command...")
            cleaned_text = listen_from_microphone(timeout=8.0, phrase_time_limit=12.0)

            if cleaned_text:
                print(f"\n[Voice Nav] Executing command: \"{cleaned_text}\"")
                process_command(cleaned_text)
            else:
                print("[Voice Nav] Could not hear a clear command. Please try again.")

            if not continuous:
                pass
            else:
                import time
                time.sleep(1.0)

        except KeyboardInterrupt:
            print("\n[Voice Nav] Exiting voice navigation.")
            break
        except EOFError:
            break
        except Exception as e:
            print(f"[Voice Nav Error]: {e}")


def main(args=None):
    if args is None:
        args = sys.argv

    # Initialize ROS 2
    rclpy.init(args=args)

    from rclpy.utilities import remove_ros_args
    clean_args = remove_ros_args(args)[1:]

    parser = argparse.ArgumentParser(description="Bangla Voice Navigation for car_bot")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file (.wav, .mp3, .mp4)")
    parser.add_argument("--continuous", action="store_true", help="Continuously listen to microphone without pressing Enter")

    parsed_opts, _ = parser.parse_known_args(clean_args)

    try:
        run_voice_navigation(audio_file=parsed_opts.audio, continuous=parsed_opts.continuous)
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
