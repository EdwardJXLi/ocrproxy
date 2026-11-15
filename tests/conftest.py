from __future__ import annotations

import pytest

from helpers import make_pdf


@pytest.fixture
def pdf_bytes() -> bytes:
    return make_pdf(2)
