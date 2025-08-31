from video_eater.__main__ import main

if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLynG1K3bq8v4yW6k3mZy5X9m5nF7H9H9D"
    main(PLAYLIST_URL,
         playlist=True,)

    print("Done!")