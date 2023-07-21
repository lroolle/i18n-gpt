# Intro

A GPT client that review i18n files(`django.po`/`translation.json`) and autofix it.

1. Grammar and typo check in related language;
2. Suggest translation to other target languages;

# Models supported

- OpenAI Chat Models, `gpt-3.5-turbo`, `gpt-4`, etc.

# Usage

```shell
poetry install

export OPENAI_API_KEY=<your_api_key>
poetry run autofix --help
```

## Review and autofix example

```shell
poetry run auotfix -w --max_msgs=100 -f examples/locale/en_US/django.po
```


# TODO

- [ ] Support more models;
- [ ]
