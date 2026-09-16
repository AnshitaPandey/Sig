"""
Text → ISL Demo
"""

from src.text_to_isl.sign_mapper import SignMapper
import time

def main():
    mapper = SignMapper()

    print("=" * 55)
    print("       Text → ISL Translation (Phase 2)")
    print("=" * 55)
    print("Type an English sentence and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        text = input("You: ").strip()

        if text.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not text:
            continue

        glosses = mapper.text_to_glosses(text)

        if not glosses:
            print("Bot: Sorry, I don't know how to sign that yet.\n")
            continue

        print(f"Glosses : {' → '.join(glosses)}")
        print(f"ISL     : {' | '.join(glosses)}")
        print()

        # Later we will play the actual sign videos / avatar here
        # For now we just show the gloss sequence

if __name__ == "__main__":
    main()