import argparse
import copy
import datetime
import itertools
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import langchain
import polib
import tiktoken
from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

now = datetime.datetime.now()
now_str = now.strftime("%Y%m%d%H%M%S")
file_handler = logging.FileHandler(filename=f"logs/{now_str}.log")
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler.setFormatter(formatter)
console_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))

logger.addHandler(file_handler)
logger.addHandler(console_handler)


# NOTE: Different languages require different prompts, especially for the shot example.
# Otherwise, the model will not be able to learn the correct translation, unless you use the gpt-4.
TARGET_LANGS = ["zh-hans", "en-us"]


# TODO: use the lanchain shot example templates
SHOT_INPUT = """
msgid "invalid format: \\"%s\\""
msgstr "invalid format: \\"%s\\""

msgid "Password"
msgstr "Password"

msgid "历史"
msgstr "历史"

msgid "The {{key}} must not include <>\\"'&:"
msgstr "The {{key}} must not include <>\\"'&:"

msgid "user_remove"
msgstr "Remote User"
"""

SHOT_OUTPUT_EN_US = """
["Invalid format: \\"%s\\"","Password","History","Please make sure that the {{key}} doesn't include any of these characters: <>\\"'&:","Remove User"]
"""

SHOT_OUTPUT_ZH_HANS = """
["格式无效：\\"%s\\"","密码","历史","请确保{{key}}不包含以下字符：<>\\"'&:","移除用户"]
"""


PROMPT_TPL = """
Act as an i18n translation machine, living in the codebase of NPM(Network Performance Management software in ITOM industry), your task is to:
- First review the msgstr is correctly correspond to the msgid;
- Try detect the language code of the msgstr, if the msgstr is not translated into {into_lang}, please translate it to {into_lang}
- Then correct any spelling, grammar, punctuation, capitalization, sentence structure, word usage, and subject-verb agreement errors in {into_lang};
- If the msgstr is correct, please just output the original msgstr;

The input POEntries which delimited by triple backticks are formatted as followings example:
Your output should be a list of msgstr values, formatted as JSON.
Please make sure that all special characters in JSON are correctly escaped by the backslash, as demonstrated below:
Input: ```{shot_input}```
Output: {shot_output}

---
POEntries: ```
{msgs}
```
"""

PROMPT_TPL_SINGLE_MSG = """
Act as an i18n translation machine, living in the codebase of NPM(Network Performance Management software in ITOM industry), your task is to:
- First review the msgstr is correctly correspond to the msgid;
-
"""


@dataclass
class GPTranslator:
    model_name: str
    api_key: str
    api_base: str
    no_cache: bool = False

    def __post_init__(self):
        self.machine = self._initialize_chat_model(
            self.model_name, self.api_key, self.api_base, self.no_cache
        )

    def _initialize_chat_model(
        self, model_name: str, api_key: str, api_base: str, no_cache: bool = False
    ):
        # TODO: Support other llm models based on langchain
        if not no_cache:
            # Cache the API result
            langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
        return ChatOpenAI(
            temperature=0.0, model_name=model_name, openai_api_base=api_base, openai_api_key=api_key
        )

    def _get_prompt(self, msgs, target_lang, prompt_tpl):
        prompt_template = ChatPromptTemplate.from_template(prompt_tpl)
        prompt_msgs = prompt_template.format_messages(
            into_lang=target_lang,
            msgs=msgs,
            shot_input=SHOT_INPUT,
            shot_output=SHOT_OUTPUT_ZH_HANS if target_lang == "zh-hans" else SHOT_OUTPUT_EN_US,
        )
        logger.debug(prompt_msgs[0].content)
        return prompt_msgs

    def review_msg(self, msg, target_lang):
        # Batch mode works fine, but somethimes failed for unexpected response from LLM.
        pass

    def review_msg_batch(self, batch, target_lang):
        tpl = PROMPT_TPL
        prompt_msgs = self._get_prompt(batch, target_lang, tpl)
        return self.machine(messages=prompt_msgs)


def read_jsonfile(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    entries = []
    for msgid, msgstr in data.items():
        entry = polib.POEntry(msgid=msgid, msgstr=msgstr)
        entries.append(entry)

    return entries


def write_new_entries_to_json(filepath, entries):
    with open(filepath, "r") as f:
        original_data = json.load(f)

    data = {entry.msgid: entry.msgstr for entry in entries}

    original_data.update(data)

    with open(filepath, "w") as f:
        json.dump(original_data, f, ensure_ascii=False, indent=4)
        logger.info(
            f"Updated: {len(entries)}/{len(original_data.values())}, written to: {filepath}."
        )


def write_entries(path, entries):
    pofile = polib.POFile()
    for entry in entries:
        pofile.append(entry)
    pofile.save(path)
    logger.info(f"Entries: {len(entries)}, Written to: {path}.")


def write_new_entries(popath, new_entries, write_path: Optional[str] = None):
    pofile = read_pofile(popath)
    new_entries_map = {entry.msgid: entry for entry in new_entries}
    for entry in pofile:
        new_entry = new_entries_map.get(entry.msgid)
        if new_entry:
            # entry.merge(new_entry)
            entry.msgstr = new_entry.msgstr
            entry.tcomment = new_entry.tcomment

    savepath = write_path or popath
    pofile.save(savepath)
    logger.info(f"Updated: {len(new_entries)}/{len(pofile)}, written to: {savepath}.")


def read_pofile(path: str) -> polib.POFile:
    return polib.pofile(path)


def clean_pofile(popath: str):
    pofile = read_pofile(popath)
    for entry in pofile:
        entry.occurrences = []
        entry.tcomment = ""

    pofile.save(popath)


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def msg_chunker(msgs, max_msgs: int = 100, max_token: int = 2000, ignore_translated: bool = False):
    num_tokens, num_msgs = 0, 0
    idx, chunk = 0, []
    original_msgs = []

    for i, entry in enumerate(msgs):
        if entry.translated() and ignore_translated:
            continue

        msgid = entry.msgid.replace('"', '\\"')
        msgstr = (entry.msgstr or entry.msgid).replace(
            '"', '\\"'
        )  # if not translated, use msgid as msgstr

        # Format the current message
        formatted_msg = f'msgid "{msgid}"\nmsgstr "{msgstr}"'

        # Calculate the number of tokens in the formatted message
        msg_tokens = num_tokens_from_string(formatted_msg)

        # If adding the current message to the current chunk would exceed the maximum
        # number of messages or tokens, yield the current chunk and the list of original messages, and start new ones
        if num_msgs + 1 > max_msgs or num_tokens + msg_tokens > max_token:
            yield idx, "\n\n".join(chunk), original_msgs
            chunk = []
            original_msgs = []
            num_tokens = 0
            num_msgs = 0

        # Add the formatted message to the current chunk and the original message to the list of original messages,
        # and update the token count and message count
        chunk.append(formatted_msg)
        original_msgs.append(entry)
        num_tokens += msg_tokens
        num_msgs += 1
        idx = i

    # Yield the last chunk and the list of original messages if they're not empty
    if chunk:
        yield idx, "\n\n".join(chunk), original_msgs


def process_response(chunk_resp, original_msgs, model_name):
    new_entries = []
    if len(chunk_resp) != len(original_msgs):
        logger.warning(
            f"Response length {len(chunk_resp)} does not match original message length {len(original_msgs)}"
        )
        return new_entries

    for processed_entry, ori_entry in zip(chunk_resp, original_msgs):
        if isinstance(processed_entry, dict):
            ori_entry.tcomment = processed_entry.get("tcomment", "")
            processed_msgstr = processed_entry.get("msgstr", ori_entry.msgstr)
        else:
            processed_msgstr = processed_entry

        new_entry = copy.deepcopy(ori_entry)
        if ori_entry.msgstr != processed_msgstr:
            new_entry.msgstr = processed_msgstr
            if not new_entry.tcomment:
                new_entry.tcomment = f"<i18n-gpt>: suggestion by {model_name}."
            new_entries.append(new_entry)
    return new_entries


def batch_review(
    translator: GPTranslator,
    entries: list,
    max_msgs: int,
    max_tokens: int,
    target_lang: str,
    total_entries: int = 0,
    retry_count: int = 0,
):
    if total_entries == 0:
        total_entries = len(entries)

    n, percent = 0, 0
    st = time.time()
    new_entries = []
    failed_entries = []
    # Iterate over the batches of messages
    for index, batch, original_msgs in msg_chunker(entries, max_msgs, max_tokens):
        n += 1
        from_linenum = original_msgs[0].linenum
        to_linenum = original_msgs[-1].linenum
        if n == 1:
            percent = 0
        else:
            # Calculate the percentage based on the index of the last entry of the current batch
            percent = round((index + 1) / total_entries * 100, 2)
        logger.info(
            f"[{percent:03.0f}%] Processing batch, idx: {index}, ln: {from_linenum}->{to_linenum}, elapsed: {time.time() - st:.2f}s"
        )
        resp = translator.review_msg_batch(batch, target_lang)
        logger.debug(resp.content)

        batch_resp = list()
        try:
            batch_resp = json.loads(resp.content)
        except Exception as e:
            logger.error(
                f"[{percent:03.0f}%] Failed to decode response, invalid JSON from llm response."
            )
            failed_entries.extend(original_msgs)
            continue

        if len(batch_resp) != len(original_msgs):
            failed_entries.extend(original_msgs)
            logger.error(
                f"[{percent:03.0f}%] Error: the number of processed msgs ({len(batch_resp)}) is not equal to the number of original msgs ({len(original_msgs)})"
            )
            continue

        new_entries.extend(process_response(batch_resp, original_msgs, translator.model_name))

    # If there are failed entries, call batch_review again with the failed entries
    if failed_entries:
        if retry_count >= 4:
            logger.error(
                f"[{percent:03.0f}%] Maximum retries reached. {len(failed_entries)} entries failed to process."
            )
            # Write the failed entries to a tmp file for review
            path = f"./tmp/{target_lang}_{time.strftime('%Y%m%d%H%M%S')}.failed.po"
            write_entries(path, failed_entries)
            return new_entries

        logger.info(f"[{percent:03.0f}%] Retrying with {len(failed_entries)} failed entries")

        # Reduce the batch size for each retry
        new_max_msgs = max(1, max_msgs // 2)
        new_max_tokens = round(max_tokens * 0.8)

        logger.info(
            f"[{percent:03.0f}%] Reducing batch size to {new_max_msgs} messages and {new_max_tokens} tokens for retry"
        )

        new_entries.extend(
            batch_review(
                translator,
                failed_entries,
                new_max_msgs,
                new_max_tokens,
                target_lang,
                total_entries,
                retry_count=retry_count + 1,
            )
        )

    return new_entries


def guess_target_lang(filepath: str) -> str:
    # Define a regular expression pattern for language codes
    pattern = r"(en[-_]us|zh[-_]hans|zh[-_]cn|en[-_]gb|fr[-_]fr|de[-_]de|ja[-_]jp|es[-_]es|it[-_]it|ru[-_]ru)"

    # Search for the pattern in the file path
    match = re.search(pattern, filepath, re.IGNORECASE)

    # If a match is found, return the matched language code with the correct format
    if match:
        return match.group(0).replace("_", "-").lower()

    # If no match is found, return a default language code
    return "en-us"


def find_files(directory, file_types):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(file_types):
                yield os.path.join(root, file)


def main():
    parser = argparse.ArgumentParser(description="i18n GPT.")
    parser.add_argument("-f", "--filepath", nargs="*", help="The path to the PO/JSON file.")
    parser.add_argument("-d", "--directory", help="The directory to search for PO/JSON files.")
    parser.add_argument(
        "-i", "--ignore_translated", action="store_true", help="Ignore translated messages."
    )
    parser.add_argument(
        "--target_lang",
        default=None,
        help="The target language (optional, can be read from the PO file path).",
    )
    parser.add_argument(
        "-w",
        "--write_file",
        action="store_true",
        help="Whether to write the corrected messages back to the PO file.",
    )
    parser.add_argument(
        "--max_msgs", type=int, default=100, help="The maximum number of messages per chunk."
    )
    parser.add_argument(
        "--max_tokens", type=int, default=2000, help="The maximum number of tokens per chunk."
    )
    parser.add_argument(
        "--model_name", default="gpt-3.5-turbo", help="The name of the language model to use."
    )
    parser.add_argument(
        "--api_key", default=os.environ.get("OPENAI_API_KEY"), help="The OpenAI API key."
    )
    parser.add_argument(
        "--api_base", default=os.environ.get("OPENAI_API_BASE_URL"), help="The OpenAI API base URL."
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="Whether to use the cache for the API calls."
    )

    args = parser.parse_args()

    dir_files = list()
    if args.directory:
        if not os.path.isdir(args.directory):
            raise ValueError(f"Directory not found: {args.directory}")
        else:
            file_types = (".po", ".json")
            for filepath in find_files(args.directory, file_types):
                dir_files.append(filepath)
        logger.info(f"Found {len(dir_files)} files in directory: {args.directory}")

    dedup_files = set()
    filepaths = args.filepath or []
    for filepath in itertools.chain(filepaths, dir_files):
        if filepath in dedup_files:
            logger.info(f"Skipping duplicate file: {filepath}")
            continue
        if not os.path.exists(filepath):
            raise ValueError(f"File not found: {filepath}")
        dedup_files.add(filepath)

        target_lang = guess_target_lang(filepath)
        if args.target_lang and target_lang != args.target_lang:
            logger.warning(
                f"Target language {args.target_lang} does not match the language code in the file path: {target_lang}"
            )
            target_lang = args.target_lang

        if not target_lang in TARGET_LANGS:
            raise ValueError(
                f"Unsupported target language: {target_lang}, supported languages: {TARGET_LANGS}"
            )

        translator = GPTranslator(
            model_name=args.model_name,
            api_key=args.api_key,
            api_base=args.api_base,
            no_cache=args.no_cache,
        )
        file_extension = os.path.splitext(filepath)[1]
        if file_extension == ".po":
            # Process PO file
            entries = read_pofile(filepath)
        elif file_extension == ".json":
            # Process JSON file
            entries = read_jsonfile(filepath)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        logger.info(
            f"Starting i18n GPT with model: {args.model_name}, target language: {target_lang}, msgs: {len(entries)}."
        )

        new_entries = batch_review(
            translator=translator,
            entries=entries,
            max_msgs=args.max_msgs,
            max_tokens=args.max_tokens,
            target_lang=target_lang,
        )

        if args.write_file:
            if file_extension == ".po":
                write_new_entries(filepath, new_entries)
            elif file_extension == ".json":
                write_new_entries_to_json(filepath, new_entries)


if __name__ == "__main__":
    main()
