import polib
import pytest

from src.main import msg_chunker, guess_target_lang


def test_msg_chunker():
    # Create a mock POFile
    pofile = polib.POFile()
    for i in range(500):
        entry = polib.POEntry(msgid=f"msgid {i}", msgstr=f"msgstr {i}")
        pofile.append(entry)

    # Test the msg_chunker function
    chunks = list(msg_chunker(pofile, max_msgs=100, max_token=2000, ignore_translated=False))

    # Check if the number of chunks is correct
    assert len(chunks) == 5

    # Check if each chunk contains the correct number of messages
    for idx, chunk, original_msgs in chunks:
        assert len(original_msgs) == 100

    # Check if the last chunk contains the correct number of messages
    _, _, last_chunk_msgs = chunks[-1]
    assert len(last_chunk_msgs) == 100

    # Test the msg_chunker function
    chunks = list(msg_chunker(pofile, max_msgs=600, max_token=10000, ignore_translated=False))

    # Check if the number of chunks is correct
    assert len(chunks) == 1


def test_msg_escaping():
    pofile = polib.POFile()
    pofile.append(
        polib.POEntry(
            msgid="The {key} must not include <>\"'&:", msgstr="The {key} must not include <>\"'&:"
        )
    )

    got = list(msg_chunker(pofile, max_msgs=1, max_token=10000, ignore_translated=False))[0][1]
    expected = '''msgid "The {key} must not include <>\\"'&:"\nmsgstr "The {key} must not include <>\\"'&:"'''

    assert got == expected, print(expected, "\n", got)


@pytest.mark.parametrize(
    "filepath, expected",
    [
        ("/path/to/en_us/file.po", "en-us"),
        ("/path/to/zh_Hans/file.po", "zh-Hans"),
        ("/path/to/EN-GB/file.po", "EN-GB"),
        ("/path/to/fr-FR/file.po", "fr-FR"),
        ("/path/to/DE_DE/file.po", "DE-DE"),
        ("/path/to/ja_JP/file.po", "ja-JP"),
        ("/path/to/ES-ES/file.po", "ES-ES"),
        ("/path/to/IT_IT/file.po", "IT-IT"),
        ("/path/to/RU-RU/file.po", "RU-RU"),
        ("extensions/smartview/grafana/public/i18n/zh-hans.json", "zh-hans"),
        ("extensions/smartview/grafana/public/i18n/en-us.json", "zh-hans"),
        ("/path/to/file.po", "en-us"),  # default case
        ("/path/to/ENUS/file.po", "en-us"),  # incorrect format, default case
    ],
)
def test_guess_target_lang(filepath, expected):
    got = guess_target_lang(filepath)
    assert got == expected, print(got)


# def test_chat(chat):
#     messages = [
#         SystemMessage(content="You are a helpful assistant that translates English to French."),
#         HumanMessage(content="Translate this sentence from English to French. I love programming."),
#     ]
#     resp = chat(messages=messages)
#     print(resp.content)
