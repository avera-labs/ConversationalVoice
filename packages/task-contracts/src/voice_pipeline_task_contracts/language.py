"""Canonical language identifiers shared by pipeline task boundaries."""

from __future__ import annotations

import re

# ISO 639-1 is deliberately embedded so accepting a language identifier does not
# depend on locale data installed on a worker image.
ISO_639_1_CODES = frozenset(
    """
    aa ab ae af ak am an ar as av ay az
    ba be bg bh bi bm bn bo br bs
    ca ce ch co cr cs cu cv cy
    da de dv dz
    ee el en eo es et eu
    fa ff fi fj fo fr fy
    ga gd gl gn gu gv
    ha he hi ho hr ht hu hy hz
    ia id ie ig ii ik io is it iu
    ja jv
    ka kg ki kj kk kl km kn ko kr ks ku kv kw ky
    la lb lg li ln lo lt lu lv
    mg mh mi mk ml mn mr ms mt my
    na nb nd ne ng nl nn no nr nv ny
    oc oj om or os
    pa pi pl ps pt
    qu
    rm rn ro ru rw
    sa sc sd se sg si sk sl sm sn so sq sr ss st su sv sw
    ta te tg th ti tk tl tn to tr ts tt tw ty
    ug uk ur uz
    ve vi vo
    wa wo
    xh yi yo
    za zh zu
    """.split()
)

_LANGUAGE_IDENTIFIER = re.compile(r"(?a:[a-z]{2}(?:-[A-Za-z0-9]{2,8})*)\Z")
_MAX_LANGUAGE_IDENTIFIER_LENGTH = 63


def parse_language_identifier(value: object) -> str:
    """Return a canonical ISO 639-1 language tag or raise ``ValueError``.

    The primary language must be a lowercase ISO 639-1 code. Optional BCP-47
    style script, region, and variant subtags are accepted so values such as
    ``zh-CN`` and ``zh-Hant-TW`` retain their original specificity.
    """
    if (
        not isinstance(value, str)
        or len(value) > _MAX_LANGUAGE_IDENTIFIER_LENGTH
        or _LANGUAGE_IDENTIFIER.fullmatch(value) is None
        or value.split("-", 1)[0] not in ISO_639_1_CODES
    ):
        raise ValueError("language must be a canonical ISO 639-1 language tag")
    return value


def primary_language(value: object) -> str:
    """Return the validated ISO 639-1 primary language subtag."""
    return parse_language_identifier(value).split("-", 1)[0]


def is_chinese_language(value: object) -> bool:
    """Return whether a validated language tag belongs to Chinese."""
    return primary_language(value) == "zh"
