from ui_helpers import scrollable_ui_layout


def test_scrollable_ui_layout_caps_height_and_scrolls_vertically():
    layout = scrollable_ui_layout()

    assert layout.width == "100%"
    assert layout.max_width == "940px"
    assert layout.max_height == "800px"
    assert layout.overflow == "hidden auto"


def test_scrollable_ui_layout_accepts_custom_bounds():
    layout = scrollable_ui_layout(max_width="700px", max_height="600px")

    assert layout.max_width == "700px"
    assert layout.max_height == "600px"
