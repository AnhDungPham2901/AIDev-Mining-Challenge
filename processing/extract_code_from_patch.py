def extract_code_from_patch(patch: str) -> str:
    """
    Convert a unified diff/patch string into clean code text.
    Removes diff headers (---, +++, @@) and keeps only added lines.
    Only keep the added code lines.
    """
    import textwrap

    lines = patch.splitlines()
    code_lines = []

    for line in lines:
        # Skip diff metadata lines
        if line.startswith(("+++", "---", "diff", "index", "@@")):
            continue
        # Keep only added code lines (remove '+')
        if line.startswith("+"):
            code_lines.append(line[1:])
        # Skip removed or unchanged lines
        else:
            continue

    # Join and strip trailing whitespace
    code = textwrap.dedent("\n".join(code_lines))
    return code
