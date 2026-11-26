"""Send a command to the running talk-to-drive simulation (sim/m4_talk_to_drive.py) and print the reply.

    python scripts/say.py "turn left"
    python scripts/say.py                 # type commands one after another (empty line or Ctrl+D to quit)
"""

import json
import sys
import urllib.request

URL = "http://127.0.0.1:8770/command"


def say(text: str) -> str:
    req = urllib.request.Request(URL, data=json.dumps({"text": text}).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())["reply"]
    except OSError as e:
        return f"(cannot reach the simulation at {URL}: {e})"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(say(" ".join(sys.argv[1:])))
    else:
        try:
            while True:
                text = input("you> ").strip()
                if not text:
                    break
                print("bot>", say(text))
        except EOFError:
            pass
