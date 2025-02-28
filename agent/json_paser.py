import re
import json



def _replace_new_line(match: re.Match[str]) -> str:
    value = match.group(2)
    value = re.sub(r"\n", r"\\n", value)
    value = re.sub(r"\r", r"\\r", value)
    value = re.sub(r"\t", r"\\t", value)
    value = re.sub(r'(?<!\\)"', r"\"", value)

    return match.group(1) + value + match.group(3)
def parse_partial_json(s: str, *, strict: bool = False):
    """Parse a JSON string that may be missing closing braces.

    Args:
        s: The JSON string to parse.
        strict: Whether to use strict parsing. Defaults to False.

    Returns:
        The parsed JSON object as a Python dictionary.
    """
    # Attempt to parse the string as-is.
    try:
        return json.loads(s, strict=strict)
    except json.JSONDecodeError:
        pass
    
def _custom_parser(multiline_string: str) -> str:
    """
    The LLM response for `action_input` may be a multiline
    string containing unescaped newlines, tabs or quotes. This function
    replaces those characters with their escaped counterparts.
    (newlines in JSON must be double-escaped: `\\n`)
    """
    if isinstance(multiline_string, (bytes, bytearray)):
        multiline_string = multiline_string.decode()

    multiline_string = re.sub(
        r'("action_input"\:\s*")(.*?)(")',
        _replace_new_line,
        multiline_string,
        flags=re.DOTALL,
    )

    return multiline_string


def parse_LLM_json(string:str)-> str:
    _json_markdown_re = re.compile(r"```(json)?(.*)", re.DOTALL)
    _json_strip_chars = " \n\r\t`"
    try:
        json_str = string.strip(_json_strip_chars)
        json_str = _custom_parser(json_str)
        json.loads(json_str, strict=True)
        # df = pd.DataFrame(json.loads(json_str, strict=True))
        return json_str
    except json.JSONDecodeError:
        json_str = _json_markdown_re.search(string)
        json_str = json_str.group(2)
        json_str = _custom_parser(json_str.strip(_json_strip_chars))
        json.loads(json_str, strict=True)
        # df = pd.DataFrame(json.loads(json_str, strict=True))
        return json_str