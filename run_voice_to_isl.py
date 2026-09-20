"""
Voice → ISL using Skeleton Avatar
"""

import speech_recognition as sr
from src.text_to_isl.sign_mapper import SignMapper
from src.text_to_isl.skeleton_avatar import SkeletonAvatar

def main():
    mapper = SignMapper()
    avatar = SkeletonAvatar()
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    print("=" * 60)
    print("     Voice → ISL  |  Skeleton Avatar")
    print("=" * 60)
    print("Speak after the prompt. Say 'quit' to stop.\n")

    with microphone as source:
        print("Calibrating microphone...")
        recognizer.adjust_for_ambient_noise(source, duration=1.5)
        print("Ready!\n")

    while True:
        try:
            print("Listening... (speak now)")
            with microphone as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)

            print("Recognizing...")
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")

            if text.lower() in ["quit", "exit", "stop"]:
                print("Goodbye!")
                break

            glosses = mapper.text_to_glosses(text)

            if not glosses:
                print("→ Sorry, I don't know how to sign that yet.\n")
                continue

            print(f"Glosses : {' → '.join(glosses)}")
            print("Avatar is signing...\n")
            avatar.play_sequence(glosses)
            print()

        except sr.WaitTimeoutError:
            print("No speech detected.\n")
        except sr.UnknownValueError:
            print("Could not understand. Try again.\n")
        except sr.RequestError as e:
            print(f"Speech error: {e}\n")
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()