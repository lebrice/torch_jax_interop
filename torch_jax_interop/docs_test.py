import textwrap

import pytest
from mktestdocs.__main__ import check_codeblock, check_raw_string

import torch_jax_interop
from torch_jax_interop.to_jax_module import torch_module_to_jax
from torch_jax_interop.to_torch_module import (
    WrappedJaxFunction,
)


def patched_grab_code_blocks(docstring: str, lang="python"):
    """Given a docstring, grab all the markdown codeblocks found in docstring.

    Patch for a bug in mkdoctest with indenting:
    - https://github.com/koaning/mktestdocs/issues/19

    Arguments:
        docstring: the docstring to analyse
        lang: if not None, the language that is assigned to the codeblock
    """
    docstring_lines = docstring.splitlines()
    docstring = (
        docstring_lines[0] + "\n" + textwrap.dedent("\n".join(docstring_lines[1:]))
    )
    in_block = False
    block = ""
    codeblocks = []
    for idx, line in enumerate(docstring.split("\n")):
        if line.strip().startswith("```"):
            if in_block:
                codeblocks.append(check_codeblock(block, lang=lang))
                block = ""
            in_block = not in_block
        if in_block:
            block += line + "\n"
    return [c for c in codeblocks if c != ""]


@pytest.mark.parametrize(
    "obj",
    [WrappedJaxFunction, torch_jax_interop, torch_module_to_jax],
    ids=lambda d: getattr(d, "__qualname__", d),
)
def test_member(obj):
    all_code = "".join(patched_grab_code_blocks(obj.__doc__, lang="python"))
    assert all_code, (obj, obj.__doc__)
    check_raw_string(all_code, lang="python")

    # assert False, mktestdocs.grab_code_blocks(WrappedJaxFunction.__doc__)
