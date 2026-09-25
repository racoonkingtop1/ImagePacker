"""ImagePacker - batch image stylization via Google Gemini or OpenAI image models."""
import base64
import json
import os
import queue
import random
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import httpx
import openai
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

import secure_store

_APPDATA = Path(os.environ.get("APPDATA", str(Path.home())))
APP_DIR = _APPDATA / "ImagePacker"
CONFIG_PATH = APP_DIR / "config.json"
# The app used to be called "NanoBananaSender"; its saved settings (including
# the DPAPI-encrypted keys) are picked up from there on first launch.
LEGACY_CONFIG_PATH = _APPDATA / "NanoBananaSender" / "config.json"

GEMINI_MODEL_PRESETS = [
    "gemini-3-pro-image",     # Nano Banana Pro - высшее качество
    "gemini-3.1-flash-image", # Nano Banana 2 - быстрее и дешевле
    "gemini-2.5-flash-image", # исходная Nano Banana
]
OPENAI_MODEL_PRESETS = [
    "gpt-image-1",       # основная модель редактирования изображений по промпту
    "gpt-image-1-mini",  # быстрее и дешевле
]
# Models that must always be selectable for Gemini, even if models.list() does
# not return them for this key (it can hide aliases / stable ids).
GEMINI_ALWAYS_LISTED = ["gemini-3-pro-image"]
MODEL_PRESETS_BY_PROVIDER = {"gemini": GEMINI_MODEL_PRESETS, "openai": OPENAI_MODEL_PRESETS}
PROVIDER_LABELS = {"gemini": "Google Gemini", "openai": "OpenAI (ChatGPT)"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

# Per-file retry policy for short-lived (per-minute) rate limiting.
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 6
MAX_BACKOFF_SECONDS = 90

# The SDK does not apply a default request timeout, so a stalled connection
# (e.g. a flaky VPN) would otherwise hang forever instead of failing with a
# clear error. Listing models is lightweight; image generation can genuinely
# take a while (especially Pro-tier models), so it gets a longer ceiling.
MODEL_LIST_TIMEOUT_MS = 20_000
GENERATE_TIMEOUT_MS = 120_000
# openai's SDK takes timeouts in seconds, not milliseconds.
MODEL_LIST_TIMEOUT_S = MODEL_LIST_TIMEOUT_MS / 1000
GENERATE_TIMEOUT_S = GENERATE_TIMEOUT_MS / 1000

DELAY_MIN, DELAY_MAX = 1, 120

# A key shorter than this can't be a real API key, so typing it never triggers a
# model-list request (Google keys are ~39 chars, OpenAI ones 50+).
MIN_KEY_LEN_FOR_AUTO_REFRESH = 20
AUTO_REFRESH_DEBOUNCE_MS = 900

# The "Готовность к запуску" row of step circles is built but hidden for now;
# flip to True to bring it back.
SHOW_READINESS = False

DEFAULT_CONFIG = {
    "provider": "gemini",
    # NOTE: API keys themselves are never stored here in plaintext - see the
    # *_api_key_enc fields below, which hold DPAPI-encrypted blobs.
    "gemini_api_key_enc": "",
    "openai_api_key_enc": "",
    "gemini_model": GEMINI_MODEL_PRESETS[0],
    "openai_model": OPENAI_MODEL_PRESETS[0],
    "prompt": "",
    "input_folder": "",
    "output_folder": "",
    "delay_seconds": 6,
}

# ---------- palette: one violet family, stepped in lightness ----------
# Every surface is a tint of the same hue (~255 deg) and the steps between them
# are small, so nothing jumps from near-black to near-white. Text is a soft
# lavender-white rather than pure white; the accent is the only saturated colour.
COLOR_BG = "#2A2738"          # page
COLOR_BAR = "#252230"         # bottom action bar (a step darker than the page)
COLOR_HEADER = "#4B3C8D"      # top bar: accent pulled toward the page colour
COLOR_LINE_DARK = "#3D3957"   # hairlines / tracks on the page
COLOR_CARD = "#38344F"        # panels (one step lighter than the page)
COLOR_FIELD = "#2C2941"       # inputs: inset, a step darker than their panel
COLOR_FIELD_LINE = "#565077"
COLOR_TEXT = "#ECE8FA"        # primary text on panels / fields
COLOR_MUTED = "#B4AED2"       # secondary text
COLOR_ACCENT = "#7455F5"
COLOR_ACCENT_DARK = "#6243E0"
COLOR_ACCENT_SOFT = "#C9BCFF"  # accent for icons / selected text on dark surfaces
COLOR_ON_ACCENT = "#FFFFFF"
COLOR_ON_DARK = "#F1EEFB"     # text sitting directly on the page
COLOR_ON_DARK_MUTED = "#B4AED2"
COLOR_DANGER = "#FF8F8F"      # stop button (sits on the bottom bar)
COLOR_DISABLED = "#7C77A0"
COLOR_HOVER_ROW = "#4A4570"
COLOR_POPUP = "#403B5E"       # dropdown list surface
COLOR_RING = "#8F88B8"        # idle step ring
# semantic text colours, tuned for the panel background
COLOR_SUCCESS = "#6EE7A8"
COLOR_ERROR = "#FF9A9A"
COLOR_WARNING = "#FFCB7A"
# the log box is the darkest surface; same semantic colours work there
LOG_OK, LOG_ERR, LOG_WARN, LOG_INFO = COLOR_SUCCESS, COLOR_ERROR, COLOR_WARNING, "#B4AED2"
COLOR_LOG_BG = "#221F2F"

FONT_H2 = ("Segoe UI", 15, "bold")
FONT_LABEL = ("Segoe UI", 9, "bold")
FONT_BODY = ("Segoe UI", 10)
FONT_HINT = ("Segoe UI", 9)
FONT_MONO = ("Consolas", 9)

# One height for every control that shares a row (inputs, dropdown, stepper,
# buttons), so neighbours line up exactly.
FIELD_H = 40


def resource_path(name: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


def load_config() -> dict:
    source = CONFIG_PATH if CONFIG_PATH.exists() else LEGACY_CONFIG_PATH
    if source.exists():
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("config root is not an object")
            merged = dict(DEFAULT_CONFIG)
            merged.update(data)
            # Migrate the old single-provider (Gemini-only) config schema.
            if data.get("api_key_enc") and not data.get("gemini_api_key_enc"):
                merged["gemini_api_key_enc"] = data["api_key_enc"]
            if data.get("model") and not data.get("gemini_model"):
                merged["gemini_model"] = data["model"]
            # Never trust/keep legacy plaintext/renamed fields left over from
            # an old version of the config file.
            for legacy_key in ("api_key", "api_key_enc", "model"):
                merged.pop(legacy_key, None)
            # A hand-edited or damaged file must not stop the app from starting.
            if merged.get("provider") not in PROVIDER_LABELS:
                merged["provider"] = DEFAULT_CONFIG["provider"]
            try:
                merged["delay_seconds"] = max(DELAY_MIN, min(DELAY_MAX, int(merged["delay_seconds"])))
            except (TypeError, ValueError):
                merged["delay_seconds"] = DEFAULT_CONFIG["delay_seconds"]
            return merged
        except (ValueError, TypeError, OSError):  # ValueError covers JSON/Unicode decode errors
            pass
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    # cfg carries convenience plaintext/alias fields for the active run
    # (see App._current_cfg) that must never be written to disk.
    skip = {"api_key", "model", "gemini_api_key", "openai_api_key"}
    to_write = {k: v for k, v in cfg.items() if k not in skip}
    CONFIG_PATH.write_text(json.dumps(to_write, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except OSError:
        pass


class _AbortBatch(Exception):
    """Raised by a per-file worker for an error that will repeat for every
    remaining file (bad key, no access to the model), so the batch stops
    instead of failing - and pausing - once per photo."""


def _is_fatal_gemini_error(e: genai_errors.APIError) -> bool:
    return getattr(e, "code", None) in (401, 403, 404) or "API_KEY_INVALID" in str(e)


def _is_quota_error(e: genai_errors.APIError) -> bool:
    return getattr(e, "code", None) == 429 or (e.status or "") == "RESOURCE_EXHAUSTED"


def _is_daily_quota_error(e: genai_errors.APIError) -> bool:
    """Best-effort detection of a per-day (vs. per-minute) quota breach, so
    the whole batch can stop instead of burning retries file after file."""
    haystack = json.dumps(e.details or {}, default=str).lower() + str(e.message or "").lower()
    return any(tok in haystack for tok in ("perday", "per day", "daily"))


def _is_openai_quota_exhausted(e: openai.RateLimitError) -> bool:
    """OpenAI uses the same 429 status for both short-lived rate limiting
    (worth retrying) and a fully exhausted billing quota (retrying is
    pointless - the account needs a top-up), distinguished only by the
    error body's `code`/message."""
    haystack = str(getattr(e, "message", "") or "").lower() + " " + str(e).lower()
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        haystack += " " + json.dumps(body, default=str).lower()
    return "insufficient_quota" in haystack or "exceeded your current quota" in haystack


# Windows virtual-key codes for the letter keys. event.keycode reports the
# physical/VK code, which stays the same regardless of the active keyboard
# layout - unlike event.keysym, which is derived from the character the
# layout produces. Tk's built-in Control-c/v/x/a bindings match on keysym,
# so under a Cyrillic (or any non-Latin) layout they silently never fire.
# Rebinding on keycode instead is the standard fix for that.
_VK_A, _VK_C, _VK_V, _VK_X = 65, 67, 86, 88


# ---------- drawing helpers ----------
def _round_rect_points(x1, y1, x2, y2, r):
    r = max(0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
    return [
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    ]


def draw_round_rect(canvas: tk.Canvas, x1, y1, x2, y2, r, **kwargs):
    return canvas.create_polygon(_round_rect_points(x1, y1, x2, y2, r), smooth=True, **kwargs)


def draw_round_rect_corners(canvas: tk.Canvas, x1, y1, x2, y2, tl, tr, br, bl, **kwargs):
    """Rounded rectangle with an independent radius per corner (0 = square)."""
    tl, tr, br, bl = (max(0, min(r, (x2 - x1) / 2, (y2 - y1) / 2)) for r in (tl, tr, br, bl))
    pts = [x1 + tl, y1, x2 - tr, y1, x2, y1, x2, y1 + tr,
           x2, y2 - br, x2, y2, x2 - br, y2, x1 + bl, y2,
           x1, y2, x1, y2 - bl, x1, y1 + tl, x1, y1]
    return canvas.create_polygon(pts, smooth=True, **kwargs)


def draw_chamfer_rect(canvas: tk.Canvas, x1, y1, x2, y2, c, **kwargs):
    """Rectangle with cut (chamfered) corners - the reference's 'tech panel'."""
    c = max(0, min(c, (x2 - x1) / 2, (y2 - y1) / 2))
    pts = [x1 + c, y1, x2 - c, y1, x2, y1 + c, x2, y2 - c,
           x2 - c, y2, x1 + c, y2, x1, y2 - c, x1, y1 + c]
    return canvas.create_polygon(pts, **kwargs)


def draw_sparkle(canvas: tk.Canvas, cx, cy, r, fill, **kwargs):
    """Four-point sparkle - the app icon's motif."""
    k = r * 0.28
    pts = [cx, cy - r, cx + k, cy - k, cx + r, cy, cx + k, cy + k,
           cx, cy + r, cx - k, cy + k, cx - r, cy, cx - k, cy - k]
    return canvas.create_polygon(pts, fill=fill, outline="", **kwargs)


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(round(c)))) for c in rgb)


def _blend(c1: str, c2: str, t: float) -> str:
    """Linearly interpolates between two hex colors (t=0 -> c1, t=1 -> c2).
    Tkinter fills have no alpha channel, so this stands in for it."""
    a, b = _hex_to_rgb(c1), _hex_to_rgb(c2)
    return _rgb_to_hex(a[i] + (b[i] - a[i]) * t for i in range(3))


def autowrap(label: tk.Label, min_width: int = 60) -> tk.Label:
    """Keeps long label text from being clipped: tracks the label's own
    allocated width (as the window/card resizes) and wraps onto it, instead
    of letting Tk silently cut the text off at the container's edge."""
    def _update(event):
        if event.width > 1:
            label.configure(wraplength=max(min_width, event.width))

    label.bind("<Configure>", _update, add="+")
    return label


def dashed_divider(parent: tk.Widget, bg=COLOR_CARD) -> tk.Canvas:
    c = tk.Canvas(parent, height=1, bg=bg, highlightthickness=0, bd=0)

    def redraw(event=None):
        c.delete("all")
        w = c.winfo_width()
        c.create_line(0, 0, w, 0, fill=COLOR_FIELD_LINE, dash=(4, 3))

    c.bind("<Configure>", redraw)
    return c


class Card(tk.Frame):
    """A lavender panel with chamfered corners drawn on a Canvas. Pack
    children into `.body`."""

    def __init__(self, parent, cut=16, bg=COLOR_CARD, pad=18):
        super().__init__(parent, bg=COLOR_BG, highlightthickness=0, bd=0)
        self.cut = cut
        self.bg_color = bg
        self.pad = pad
        self._bg_id = None

        self.canvas = tk.Canvas(self, bg=COLOR_BG, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        self.body = tk.Frame(self.canvas, bg=bg)
        self._win = self.canvas.create_window(pad, pad, window=self.body, anchor="nw")

        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.body.bind("<Configure>", self._on_body_resize)

    def _on_canvas_resize(self, event):
        w, h = event.width, event.height
        self.canvas.coords(self._win, self.pad, self.pad)
        self.canvas.itemconfig(self._win, width=max(0, w - 2 * self.pad))
        self._redraw_bg(w, h)

    def _on_body_resize(self, event):
        self.canvas.configure(height=event.height + 2 * self.pad)

    def _redraw_bg(self, w, h):
        if self._bg_id:
            self.canvas.delete(self._bg_id)
            self._bg_id = None
        if w > 2 and h > 2:
            self._bg_id = draw_chamfer_rect(
                self.canvas, 0, 0, w, h, self.cut, fill=self.bg_color, outline="",
            )
            self.canvas.tag_lower(self._bg_id)


class ScrollableFrame(tk.Frame):
    """Makes its `.inner` frame vertically scrollable once content overflows
    the window - so nothing gets clipped if the window is resized smaller
    than the content, or on a lower-resolution screen."""

    def __init__(self, parent):
        super().__init__(parent, bg=COLOR_BG)
        self.canvas = tk.Canvas(self, bg=COLOR_BG, highlightthickness=0, bd=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vbar.pack(side="right", fill="y")

        self.inner = tk.Frame(self.canvas, bg=COLOR_BG)
        self._win = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")

    def _on_inner_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self._win, width=event.width)

    def _on_mousewheel(self, event):
        # Let text boxes (prompt, log) scroll themselves; only the page
        # background scrolls the outer canvas.
        if isinstance(event.widget, tk.Text):
            return
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class PillButton(tk.Canvas):
    """A pill-shaped, hover/press-reactive button drawn on a Canvas, since
    ttk cannot round its corners.

    Variants: "accent" (filled violet - the primary action) and "danger"
    (outlined, for stopping) - the reference's filled and outlined chips.

    Supports the small subset of ttk.Button's API this app relies on
    (`.configure(state=...)`, `.configure(text=...)`) so it's a drop-in
    replacement at call sites.
    """

    _VARIANTS = {
        "accent": dict(fill=COLOR_ACCENT, text=COLOR_ON_ACCENT, border=None),
        "danger": dict(fill=None, text=COLOR_DANGER, border=COLOR_DANGER),
    }

    def __init__(self, parent, text, command=None, variant="accent", bg=COLOR_CARD,
                 width=None, height=FIELD_H, font=("Segoe UI", 10, "bold"), padx=20):
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0, height=height,
                          cursor="hand2")
        self._spec = self._VARIANTS[variant]
        self._variant = variant
        self._surface = bg
        self._text = text
        self._command = command
        weight = font[2] if len(font) > 2 else "normal"
        self._font = tkfont.Font(family=font[0], size=font[1], weight=weight)
        self._enabled = True
        self._hover = False
        self._pressed = False
        self._bh = height
        self._bw = width or (self._font.measure(text) + padx * 2)
        self.configure(width=self._bw, height=self._bh)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self._redraw()

    def _colors(self):
        spec, surface = self._spec, self._surface
        base_fill = spec["fill"] or surface
        if not self._enabled:
            muted = _blend(surface, COLOR_DISABLED, 0.55)
            if spec["fill"]:
                return muted, None, COLOR_ON_ACCENT
            return surface, muted, muted
        if self._pressed:
            fill = (_blend(base_fill, "#000000", 0.22) if spec["fill"]
                    else _blend(surface, spec["border"], 0.28))
        elif self._hover:
            fill = (_blend(base_fill, "#FFFFFF", 0.14) if spec["fill"]
                    else _blend(surface, spec["border"], 0.14))
        else:
            fill = base_fill
        return fill, spec["border"], spec["text"]

    def _redraw(self):
        self.delete("all")
        w, h = self._bw, self._bh
        fill, outline, text_color = self._colors()
        draw_round_rect(self, 1, 1, w - 1, h - 1, h / 2, fill=fill,
                         outline=outline or "", width=1.6 if outline else 0)
        yoff = 1 if self._pressed else 0
        self.create_text(w / 2, h / 2 + yoff, text=self._text, fill=text_color, font=self._font)

    def _on_enter(self, _e):
        if self._enabled:
            self._hover = True
            self._redraw()

    def _on_leave(self, _e):
        self._hover = False
        self._pressed = False
        self._redraw()

    def _on_press(self, _e):
        if self._enabled:
            self._pressed = True
            self._redraw()

    def _on_release(self, e):
        was_pressed = self._pressed
        self._pressed = False
        self._redraw()
        inside = 0 <= e.x <= self._bw and 0 <= e.y <= self._bh
        if self._enabled and was_pressed and inside and self._command:
            self._command()

    def configure(self, **kwargs):
        redraw = False
        if "state" in kwargs:
            self._enabled = kwargs.pop("state") != "disabled"
            if not self._enabled:
                self._hover = self._pressed = False
            redraw = True
        if "text" in kwargs:
            self._text = kwargs.pop("text")
            redraw = True
        if kwargs:
            super().configure(**kwargs)
        if redraw:
            self._redraw()

    config = configure


class InputField(tk.Canvas):
    """A pill-shaped text input: a borderless tk.Entry embedded in a rounded
    canvas, with an optional trailing icon INSIDE the field ("eye" for
    secrets, "chevron" for dropdowns). Being drawn on one canvas, every field
    gets exactly the same height as the buttons beside it."""

    PAD = 18
    TRAIL_W = 52
    # Some touchpads/mice deliver a tap as two presses in quick succession; for a
    # toggle that would flip the state twice and look like "nothing happened".
    TRAIL_DEBOUNCE_S = 0.3

    def __init__(self, parent, textvariable, bg=COLOR_CARD, show="", trailing=None,
                 on_trailing=None, height=FIELD_H, justify="left"):
        # A small requested width lets the field stretch (pack fill/expand)
        # without ever pushing the button beside it out of the row.
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0, height=height, width=60)
        self._bh = height
        self._focused = False
        self._trailing = trailing
        self._on_trailing = on_trailing
        self._trail_on = False
        self._trail_hover = False
        self._last_trail_click = 0.0
        self.entry = tk.Entry(
            self, textvariable=textvariable, show=show, bd=0, relief="flat",
            highlightthickness=0, bg=COLOR_FIELD, fg=COLOR_TEXT, insertbackground=COLOR_TEXT,
            font=FONT_BODY, selectbackground=COLOR_ACCENT, selectforeground=COLOR_ON_ACCENT,
            justify=justify,
        )
        self._win = self.create_window(self.PAD, height / 2, window=self.entry, anchor="w")
        self.bind("<Configure>", lambda _e: self._layout())
        self.bind("<Button-1>", self._on_click)
        self.bind("<Motion>", self._on_motion)
        self.bind("<Leave>", self._on_leave)
        self.entry.bind("<FocusIn>", lambda _e: self._set_focus(True))
        self.entry.bind("<FocusOut>", lambda _e: self._set_focus(False))

    def set_icon_state(self, on: bool):
        self._trail_on = on
        self._redraw()

    def _set_focus(self, focused: bool):
        self._focused = focused
        self._redraw()

    def _icon_center(self):
        return self.winfo_width() - self.TRAIL_W / 2 - 2, self._bh / 2

    def _over_icon(self, x):
        return bool(self._trailing) and x >= self.winfo_width() - self.TRAIL_W

    def _on_click(self, e):
        if self._over_icon(e.x):
            now = time.monotonic()
            if self._on_trailing and now - self._last_trail_click >= self.TRAIL_DEBOUNCE_S:
                self._last_trail_click = now
                self._on_trailing()
        else:
            self.entry.focus_set()

    def _on_motion(self, e):
        hover = self._over_icon(e.x)
        if hover != self._trail_hover:
            self._trail_hover = hover
            self.configure(cursor="hand2" if hover else "")
            self._redraw()

    def _on_leave(self, _e):
        if self._trail_hover:
            self._trail_hover = False
            self.configure(cursor="")
            self._redraw()

    def _layout(self):
        w = self.winfo_width()
        right = self.TRAIL_W if self._trailing else self.PAD
        self.coords(self._win, self.PAD, self._bh / 2)
        self.itemconfig(self._win, width=max(10, w - self.PAD - right))
        self._redraw()

    def _redraw(self):
        self.delete("shape", "icon")
        w, h = self.winfo_width(), self._bh
        if w <= 2:
            return
        line = COLOR_ACCENT if (self._focused or self._trail_on) else COLOR_FIELD_LINE
        draw_round_rect(self, 1, 1, w - 1, h - 1, h / 2, fill=COLOR_FIELD, outline=line,
                         width=2 if self._focused else 1.5, tags="shape")
        self.tag_lower("shape")
        if self._trailing == "eye":
            self._draw_eye(*self._icon_center())
        elif self._trailing == "chevron":
            self._draw_chevron(*self._icon_center())

    def _draw_eye(self, cx, cy):
        if self._trail_hover:
            self.create_oval(cx - 15, cy - 15, cx + 15, cy + 15, fill=COLOR_HOVER_ROW, outline="",
                             tags="icon")
        col = COLOR_ACCENT_SOFT if self._trail_on else COLOR_MUTED
        almond = [-11, 0, -6, -4.6, 0, -6.4, 6, -4.6, 11, 0, 6, 4.6, 0, 6.4, -6, 4.6]
        pts = []
        for i in range(0, len(almond), 2):
            pts += [cx + almond[i], cy + almond[i + 1]]
        self.create_polygon(pts, smooth=True, fill="", outline=col, width=1.8, tags="icon")
        self.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill=col, outline="", tags="icon")
        if not self._trail_on:  # crossed-out eye = value is hidden
            self.create_line(cx - 9, cy + 8, cx + 9, cy - 8, fill=col, width=2, capstyle="round",
                             tags="icon")

    def _draw_chevron(self, cx, cy):
        if self._trail_hover:
            self.create_oval(cx - 15, cy - 15, cx + 15, cy + 15, fill=COLOR_HOVER_ROW, outline="",
                             tags="icon")
        col = COLOR_ACCENT_SOFT if self._trail_on else COLOR_MUTED
        d = -1 if self._trail_on else 1
        self.create_line(cx - 5, cy - 2 * d, cx, cy + 3 * d, cx + 5, cy - 2 * d, fill=col,
                         width=2, capstyle="round", joinstyle="round", tags="icon")


class ComboField(InputField):
    """An editable dropdown: the field itself takes typed text, the chevron
    opens a styled list. Replaces the native (unstylable) ttk.Combobox."""

    ROW_H = 38
    MAX_ROWS = 7

    def __init__(self, parent, textvariable, values, **kwargs):
        super().__init__(parent, textvariable, trailing="chevron",
                         on_trailing=self.toggle_popup, **kwargs)
        self._var = textvariable
        self._values = list(values)
        self._popup = None
        self._root_click_id = None
        self.entry.bind("<Down>", lambda _e: self._open_popup())

    def set_values(self, values):
        self._values = list(values)
        self._close_popup()

    def toggle_popup(self):
        if self._popup:
            self._close_popup()
        else:
            self._open_popup()

    def _open_popup(self):
        if self._popup or not self._values:
            return
        top = self.winfo_toplevel()
        self.update_idletasks()
        w = self.winfo_width()
        vis = min(len(self._values), self.MAX_ROWS)
        h = vis * self.ROW_H + 16
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height() + 4
        if y + h > self.winfo_screenheight() - 8:
            y = self.winfo_rooty() - h - 4

        key = "#010203"  # colour keyed out below -> rounded corners on a Toplevel
        pop = tk.Toplevel(top)
        pop.overrideredirect(True)
        pop.attributes("-topmost", True)
        pop.configure(bg=key)
        try:
            pop.attributes("-transparentcolor", key)
        except tk.TclError:
            pass
        pop.geometry(f"{w}x{h}+{x}+{y}")
        cv = tk.Canvas(pop, width=w, height=h, bg=key, highlightthickness=0, bd=0)
        cv.pack()
        state = {"offset": 0, "hover": None}
        current = self._var.get()
        row_font = tkfont.Font(family="Segoe UI", size=10)
        row_bold = tkfont.Font(family="Segoe UI", size=10, weight="bold")

        def render():
            cv.delete("all")
            draw_round_rect(cv, 1, 1, w - 1, h - 1, 20, fill=COLOR_POPUP, outline=COLOR_ACCENT,
                             width=2)
            for i in range(vis):
                idx = state["offset"] + i
                if idx >= len(self._values):
                    break
                y0 = 8 + i * self.ROW_H
                value = self._values[idx]
                selected = value == current
                if state["hover"] == idx:
                    draw_round_rect(cv, 8, y0 + 1, w - 8, y0 + self.ROW_H - 1, 14,
                                     fill=COLOR_HOVER_ROW, outline="")
                cv.create_text(22, y0 + self.ROW_H / 2, text=value, anchor="w",
                               fill=COLOR_ACCENT_SOFT if selected else COLOR_TEXT,
                               font=row_bold if selected else row_font)
                if selected:
                    cv.create_text(w - 24, y0 + self.ROW_H / 2, text="✓", anchor="e",
                                   fill=COLOR_ACCENT_SOFT, font=("Segoe UI Symbol", 11, "bold"))

        def index_at(ev_y):
            i = (ev_y - 8) // self.ROW_H
            if 0 <= i < vis and state["offset"] + i < len(self._values):
                return state["offset"] + i
            return None

        def on_motion(ev):
            hover = index_at(ev.y)
            if hover != state["hover"]:
                state["hover"] = hover
                render()

        def on_leave(_ev):
            state["hover"] = None
            render()

        def on_click(ev):
            idx = index_at(ev.y)
            if idx is not None:
                self._var.set(self._values[idx])
                self._close_popup()

        def on_wheel(ev):
            max_off = max(0, len(self._values) - vis)
            state["offset"] = max(0, min(max_off, state["offset"] + (-1 if ev.delta > 0 else 1)))
            render()
            return "break"  # otherwise the page behind the list scrolls too

        cv.bind("<Motion>", on_motion)
        cv.bind("<Leave>", on_leave)
        cv.bind("<Button-1>", on_click)
        cv.bind("<MouseWheel>", on_wheel)
        pop.bind("<Escape>", lambda _e: self._close_popup())
        render()
        self._popup = pop
        self.set_icon_state(True)
        self._root_click_id = top.bind("<Button-1>", self._on_root_click, add="+")

    def _on_root_click(self, event):
        # Clicks on the field itself are handled by toggle_popup; anything else
        # outside the list dismisses it.
        if event.widget is not self:
            self._close_popup()

    def _close_popup(self):
        if self._root_click_id:
            try:
                self.winfo_toplevel().unbind("<Button-1>", self._root_click_id)
            except tk.TclError:
                pass
            self._root_click_id = None
        if self._popup:
            try:
                self._popup.destroy()
            except tk.TclError:
                pass
            self._popup = None
        self.set_icon_state(False)


class StepperField(tk.Canvas):
    """A numeric stepper: [ - | 6 | + ] - three equal cells filling one pill,
    the '-' and '+' cells sitting flush against the container's edges. Replaces
    the native (unstylable) ttk.Spinbox."""

    CELL = 36
    WIDTH = CELL * 3
    HEIGHT = 30  # deliberately compact - this is a secondary setting

    def __init__(self, parent, variable: tk.IntVar, minimum=DELAY_MIN, maximum=DELAY_MAX,
                 bg=COLOR_CARD, height=HEIGHT):
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0, height=height,
                          width=self.WIDTH)
        self.var, self._min, self._max = variable, minimum, maximum
        self._bh = height
        self._focused = False
        self._hover = None  # "minus" | "plus" | None
        self._repeat_id = None
        self._text = tk.StringVar(value=str(self._clamp(variable.get())))
        self.entry = tk.Entry(
            self, textvariable=self._text, width=4, bd=0, relief="flat", highlightthickness=0,
            bg=COLOR_FIELD, fg=COLOR_TEXT, insertbackground=COLOR_TEXT, font=("Segoe UI", 10, "bold"),
            justify="center", selectbackground=COLOR_ACCENT, selectforeground=COLOR_ON_ACCENT,
        )
        self.create_window(self.WIDTH / 2, height / 2, window=self.entry, anchor="center",
                           width=self.CELL - 8)
        self.entry.bind("<KeyRelease>", self._on_key)
        self.entry.bind("<FocusIn>", lambda _e: self._set_focus(True))
        self.entry.bind("<FocusOut>", self._on_focus_out)
        for widget in (self, self.entry):
            widget.bind("<MouseWheel>", self._on_wheel)
        self.bind("<Motion>", self._on_motion)
        self.bind("<Leave>", lambda _e: self._set_hover(None))
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self._redraw()

    def _clamp(self, v):
        return max(self._min, min(self._max, int(v)))

    def _set_focus(self, focused):
        self._focused = focused
        self._redraw()

    def _hit(self, x, _y):
        if x <= self.CELL:
            return "minus"
        if x >= self.WIDTH - self.CELL:
            return "plus"
        return None

    def _redraw(self):
        self.delete("shape")
        w, h = self.WIDTH, self._bh
        line = COLOR_ACCENT if self._focused else COLOR_FIELD_LINE
        draw_round_rect(self, 1, 1, w - 1, h - 1, h / 2, fill=COLOR_FIELD, outline=line,
                         width=2 if self._focused else 1.5, tags="shape")
        # The end cells are exactly as wide as the number cell and flush with the
        # container edge: only their outer corners are rounded.
        r = h / 2 - 1
        cells = (("minus", 1, self.CELL, (r, 0, 0, r)),
                 ("plus", w - self.CELL, w - 1, (0, r, r, 0)))
        for which, x1, x2, (tl, tr, br, bl) in cells:
            fill = COLOR_ACCENT_DARK if self._hover == which else COLOR_ACCENT
            draw_round_rect_corners(self, x1, 1, x2, h - 1, tl, tr, br, bl, fill=fill, outline="",
                                    tags="shape")
            cx, cy = (x1 + x2) / 2, h / 2
            self.create_line(cx - 4.5, cy, cx + 4.5, cy, fill=COLOR_ON_ACCENT, width=2,
                             capstyle="round", tags="shape")
            if which == "plus":
                self.create_line(cx, cy - 4.5, cx, cy + 4.5, fill=COLOR_ON_ACCENT, width=2,
                                 capstyle="round", tags="shape")
        self.tag_lower("shape")

    def _set_hover(self, which):
        if which != self._hover:
            self._hover = which
            self.configure(cursor="hand2" if which else "")
            self._redraw()

    def _on_motion(self, e):
        self._set_hover(self._hit(e.x, e.y))

    def _step(self, delta):
        value = self._clamp(self.var.get() + delta)
        self.var.set(value)
        self._text.set(str(value))

    def _on_press(self, e):
        which = self._hit(e.x, e.y)
        if not which:
            self.entry.focus_set()
            return
        delta = -1 if which == "minus" else 1
        self._step(delta)
        self._repeat_id = self.after(420, lambda: self._repeat(delta))

    def _repeat(self, delta):
        self._step(delta)
        self._repeat_id = self.after(70, lambda: self._repeat(delta))

    def _on_release(self, _e):
        if self._repeat_id:
            self.after_cancel(self._repeat_id)
            self._repeat_id = None

    def _on_wheel(self, e):
        self._step(1 if e.delta > 0 else -1)
        return "break"

    def _on_key(self, _e):
        text = self._text.get().strip()
        if text.isdigit() and self._min <= int(text) <= self._max:
            self.var.set(int(text))

    def _on_focus_out(self, _e):
        self._set_focus(False)
        text = self._text.get().strip()
        value = self._clamp(text) if text.isdigit() else self.var.get()
        self.var.set(value)
        self._text.set(str(value))


class TextBox(tk.Canvas):
    """A multi-line text area in a rounded box (optionally with a slim
    scrollbar), so it matches the pill inputs instead of a raw tk.Text."""

    PAD = 14

    def __init__(self, parent, lines=5, bg=COLOR_CARD, fill=COLOR_FIELD, fg=COLOR_TEXT,
                 outline=COLOR_FIELD_LINE, focus_outline=COLOR_ACCENT, font=FONT_BODY,
                 scrollbar=False, readonly=False, radius=20):
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0)
        self._fill, self._outline, self._focus_outline = fill, outline, focus_outline
        self._radius = radius
        self._focused = False
        self._has_bar = scrollbar
        self.text = tk.Text(
            self, height=lines, wrap="word", bg=fill, fg=fg, insertbackground=fg, relief="flat",
            bd=0, highlightthickness=0, padx=0, pady=0, font=font,
            selectbackground=COLOR_ACCENT, selectforeground=COLOR_ON_ACCENT,
            state="disabled" if readonly else "normal",
        )
        self._text_win = self.create_window(self.PAD + 2, self.PAD, window=self.text, anchor="nw")
        self._bar = None
        if scrollbar:
            self._bar = ttk.Scrollbar(self, orient="vertical", command=self.text.yview,
                                      style="Box.Vertical.TScrollbar")
            self.text.configure(yscrollcommand=self._bar.set)
            self._bar_win = self.create_window(0, self.PAD, window=self._bar, anchor="ne")
        self.update_idletasks()
        self._inner_h = self.text.winfo_reqheight()
        self.configure(height=self._inner_h + 2 * self.PAD)
        self.bind("<Configure>", lambda _e: self._layout())
        self.text.bind("<FocusIn>", lambda _e: self._set_focus(True))
        self.text.bind("<FocusOut>", lambda _e: self._set_focus(False))

    def _set_focus(self, focused):
        self._focused = focused
        self._redraw()

    def _layout(self):
        w = self.winfo_width()
        bar_w = 14 if self._bar else 0
        self.itemconfig(self._text_win, width=max(10, w - 2 * self.PAD - 4 - bar_w),
                        height=self._inner_h)
        if self._bar:
            self.coords(self._bar_win, w - self.PAD + 4, self.PAD)
            self.itemconfig(self._bar_win, height=self._inner_h)
        self._redraw()

    def _redraw(self):
        self.delete("shape")
        w, h = self.winfo_width(), self.winfo_height()
        if w <= 2 or h <= 2:
            return
        draw_round_rect(self, 1, 1, w - 1, h - 1, self._radius, fill=self._fill,
                         outline=self._focus_outline if self._focused else self._outline,
                         width=2 if self._focused else 1.5, tags="shape")
        self.tag_lower("shape")


class SegmentedControl(tk.Canvas):
    """One rounded container holding every choice, with the selected segment
    filled - so it's obvious the options belong together and one must be picked.
    Stretches to the full width of its parent, segments sharing it equally."""

    def __init__(self, parent, options, variable: tk.StringVar, command=None, bg=COLOR_CARD,
                 height=46, font=("Segoe UI", 10, "bold")):
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0, height=height,
                          width=60, cursor="hand2")
        self.options = list(options)  # [(value, label), ...]
        self.var = variable
        self._command = command
        weight = font[2] if len(font) > 2 else "normal"
        self._font = tkfont.Font(family=font[0], size=font[1], weight=weight)
        self._bh = height
        self._pad = 4
        self._thumb_x = None  # animated left edge of the selected segment
        self._anim_id = None
        self.bind("<Configure>", lambda _e: self._sync_thumb(animate=False))
        self.bind("<Button-1>", self._on_click)

    def _seg_w(self):
        return (self.winfo_width() - 2 * self._pad) / max(1, len(self.options))

    def _index(self):
        for i, (value, _label) in enumerate(self.options):
            if value == self.var.get():
                return i
        return 0

    def _sync_thumb(self, animate=True):
        target = self._pad + self._index() * self._seg_w()
        if self._anim_id:
            self.after_cancel(self._anim_id)
            self._anim_id = None
        if not animate or self._thumb_x is None:
            self._thumb_x = target
            self._redraw()
        else:
            self._step_to(target)

    def _step_to(self, target):
        self._thumb_x += (target - self._thumb_x) * 0.4
        if abs(target - self._thumb_x) < 1:
            self._thumb_x = target
            self._anim_id = None
        else:
            self._anim_id = self.after(16, lambda: self._step_to(target))
        self._redraw()

    def _redraw(self):
        self.delete("all")
        w, h, pad = self.winfo_width(), self._bh, self._pad
        if w <= 2 or self._thumb_x is None:
            return
        draw_round_rect(self, 1, 1, w - 1, h - 1, h / 2, fill=COLOR_FIELD,
                         outline=COLOR_FIELD_LINE, width=1.5)
        seg = self._seg_w()
        draw_round_rect(self, self._thumb_x, pad, self._thumb_x + seg, h - pad,
                         (h - 2 * pad) / 2, fill=COLOR_ACCENT, outline="")
        for i, (value, label) in enumerate(self.options):
            active = value == self.var.get()
            self.create_text(pad + seg * (i + 0.5), h / 2, text=label, font=self._font,
                             fill=COLOR_ON_ACCENT if active else COLOR_MUTED)

    def _on_click(self, e):
        idx = int((e.x - self._pad) // max(1, self._seg_w()))
        idx = max(0, min(len(self.options) - 1, idx))
        value = self.options[idx][0]
        if value != self.var.get():
            self.var.set(value)
            self._sync_thumb()
            if self._command:
                self._command()


class HeaderBar(tk.Canvas):
    """A compact site-style top bar: logo mark and a light+bold two-weight
    wordmark on the left (like 'Global / Tech' in the reference), a short
    descriptor on the right, and a little circuit-trace decoration."""

    HEIGHT = 64

    def __init__(self, parent):
        super().__init__(parent, bg=COLOR_HEADER, highlightthickness=0, bd=0, height=self.HEIGHT)
        self._light = tkfont.Font(family="Segoe UI Light", size=20)
        self._bold = tkfont.Font(family="Segoe UI", size=20, weight="bold")
        self.bind("<Configure>", lambda _e: self._layout())

    def _layout(self):
        w, h = self.winfo_width(), self.HEIGHT
        if w < 50:
            return
        self.delete("all")
        soft = _blend(COLOR_HEADER, "#FFFFFF", 0.07)
        line = _blend(COLOR_HEADER, "#FFFFFF", 0.22)
        # soft diagonal light streak on the right, echoing the reference's gradient panels
        self.create_polygon(w * 0.78, 0, w, 0, w, h, w * 0.72, h, fill=soft, outline="")
        # one circuit trace: run, 45-degree jog, run, end node
        y = h * 0.62
        pts = [w * 0.42, y, w * 0.47, y, w * 0.49, y - 12, w * 0.55, y - 12]
        self.create_line(*pts, fill=line, width=1.5)
        self.create_oval(pts[-2] - 3, pts[-1] - 3, pts[-2] + 3, pts[-1] + 3, outline=line, width=1.5)
        draw_sparkle(self, 34, h / 2, 12, COLOR_ON_DARK)
        wl = self._light.measure("Image")
        self.create_text(58, h / 2, text="Image", anchor="w", font=self._light, fill=COLOR_ON_DARK)
        self.create_text(58 + wl, h / 2, text="Packer", anchor="w", font=self._bold,
                         fill=COLOR_ON_DARK)
        if w >= 520:
            self.create_text(w - 24, h / 2, text="Пакетная ИИ-обработка фото", anchor="e",
                             font=("Segoe UI", 10), fill=COLOR_ON_DARK_MUTED)
        self.create_line(0, h - 1, w, h - 1, fill=_blend(COLOR_HEADER, "#000000", 0.25))


class RoundedProgressBar(tk.Canvas):
    """A pill-shaped progress bar - ttk.Progressbar cannot round its ends.
    Exposes the same `.configure(maximum=..., value=...)` surface the app's
    batch loop already calls, so it drops in without touching that logic.
    """

    def __init__(self, parent, bg=COLOR_BAR, height=10):
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0, height=height)
        self._bh = height
        self._maximum = 100
        self._value = 0
        self.bind("<Configure>", lambda _e: self._redraw())
        self._redraw()

    def configure(self, **kwargs):
        redraw = False
        if "maximum" in kwargs:
            self._maximum = max(1, kwargs.pop("maximum"))
            redraw = True
        if "value" in kwargs:
            self._value = kwargs.pop("value")
            redraw = True
        if kwargs:
            super().configure(**kwargs)
        if redraw:
            self._redraw()

    config = configure

    def _redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self._bh
        if w <= 2:
            return
        draw_round_rect(self, 0, 0, w, h, h / 2, fill=COLOR_LINE_DARK, outline="")
        frac = 0.0 if self._maximum <= 0 else max(0.0, min(1.0, self._value / self._maximum))
        fill_w = w * frac
        if fill_w >= 1:
            fill_w = max(fill_w, h)  # keep the rounded cap from looking clipped near 0%
            draw_round_rect(self, 0, 0, fill_w, h, h / 2, fill=COLOR_ACCENT, outline="")


class StepBadge(tk.Canvas):
    """A ringed circle: shows the step number while pending and a check on a
    filled violet circle once satisfied."""

    def __init__(self, parent, number, bg=COLOR_BG, size=54):
        super().__init__(parent, bg=bg, width=size, height=size, highlightthickness=0, bd=0)
        self._number = number
        self._size = size
        self._done = None
        self.set_done(False)

    def set_done(self, done: bool):
        if done == self._done:
            return
        self._done = done
        self.delete("all")
        s, m = self._size, 3
        if done:
            self.create_oval(m, m, s - m, s - m, fill=COLOR_ACCENT, outline=COLOR_ACCENT, width=2)
            self.create_text(s / 2, s / 2, text="✓", fill=COLOR_ON_ACCENT,
                              font=("Segoe UI Symbol", 16, "bold"))
        else:
            self.create_oval(m, m, s - m, s - m, fill=COLOR_BG, outline=COLOR_RING, width=2)
            self.create_text(s / 2, s / 2, text=str(self._number), fill=COLOR_ON_DARK,
                              font=("Segoe UI", 15, "bold"))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ImagePacker")
        # Leave room for the taskbar/title bar so the fixed bottom bar (Start
        # button) is never pushed off-screen on smaller displays.
        height = max(540, min(860, self.winfo_screenheight() - 230))
        self.geometry(f"620x{height}")
        self.minsize(500, 520)
        self.configure(bg=COLOR_BG)
        try:
            self.iconbitmap(resource_path("icon.ico"))
        except tk.TclError:
            pass

        self.cfg = load_config()
        self._keys = {
            "gemini": secure_store.decrypt(self.cfg.get("gemini_api_key_enc", "")),
            "openai": secure_store.decrypt(self.cfg.get("openai_api_key_enc", "")),
        }
        self._models = {
            "gemini": self.cfg.get("gemini_model") or GEMINI_MODEL_PRESETS[0],
            "openai": self.cfg.get("openai_model") or OPENAI_MODEL_PRESETS[0],
        }
        self._active_provider = self.cfg.get("provider") or "gemini"
        self.log_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self._models_loading = False
        self._auto_refresh_job = None
        # The key we last auto-fetched the model list for, per provider - so the
        # same key is never fetched twice on its own (the button always can).
        self._auto_tried: dict[str, str] = {}
        # The last model list fetched per provider, so switching providers back and
        # forth doesn't throw the fetched list away (the key is not re-fetched).
        self._fetched_models: dict[str, list[str]] = {}
        self._dup_stems: set[str] = set()

        self._apply_theme()
        self._build_ui()
        self._fix_clipboard_shortcuts()
        self.after(100, self._poll_log_queue)
        if SHOW_READINESS:
            self.after(200, self._steps_tick)
        # If a key is already saved, load the model list as soon as the window is up.
        self.after(500, self._auto_refresh_models)
        self.var_api_key.trace_add("write", self._on_key_changed)

    # ---------- theme ----------
    def _apply_theme(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        # Only scrollbars remain native; every other control is custom-drawn.
        style.configure("Vertical.TScrollbar", background=COLOR_LINE_DARK, troughcolor=COLOR_BG,
                         bordercolor=COLOR_BG, lightcolor=COLOR_LINE_DARK, darkcolor=COLOR_LINE_DARK,
                         arrowcolor=COLOR_ON_DARK_MUTED, gripcount=0)
        style.map("Vertical.TScrollbar", background=[("active", COLOR_ACCENT)])
        style.configure("Box.Vertical.TScrollbar", background=COLOR_LINE_DARK, troughcolor=COLOR_LOG_BG,
                         bordercolor=COLOR_LOG_BG, lightcolor=COLOR_LINE_DARK, darkcolor=COLOR_LINE_DARK,
                         arrowcolor=COLOR_ON_DARK_MUTED, gripcount=0)
        style.map("Box.Vertical.TScrollbar", background=[("active", COLOR_ACCENT)])

    # ---------- small UI builders ----------
    def _row(self, parent, bg=COLOR_CARD):
        return tk.Frame(parent, bg=bg)

    def _label(self, parent, text, bg=COLOR_CARD):
        return autowrap(tk.Label(parent, text=text, bg=bg, fg=COLOR_TEXT, font=FONT_LABEL,
                                  anchor="w", justify="left"))

    def _label_var(self, parent, var, bg=COLOR_CARD):
        return autowrap(tk.Label(parent, textvariable=var, bg=bg, fg=COLOR_TEXT, font=FONT_LABEL,
                                  anchor="w", justify="left"))

    def _hint(self, parent, text, bg=COLOR_CARD, fg=COLOR_MUTED):
        return autowrap(tk.Label(parent, text=text, bg=bg, fg=fg, font=FONT_HINT,
                                  anchor="w", justify="left"))

    def _hint_var(self, parent, var, bg=COLOR_CARD):
        return autowrap(tk.Label(parent, textvariable=var, bg=bg, fg=COLOR_MUTED, font=FONT_HINT,
                                  anchor="w", justify="left"))

    def _section(self, parent, title, sub=None):
        """A bold heading with a muted one-line subhead."""
        wrap = tk.Frame(parent, bg=COLOR_BG)
        wrap.pack(fill="x", padx=20, pady=(24, 10))
        tk.Label(wrap, text=title, bg=COLOR_BG, fg=COLOR_ON_DARK, font=FONT_H2,
                 anchor="w").pack(fill="x")
        if sub:
            self._hint(wrap, sub, bg=COLOR_BG, fg=COLOR_ON_DARK_MUTED).pack(fill="x", pady=(2, 0))

    def _build_bottom_bar(self):
        bar = tk.Frame(self, bg=COLOR_BAR)
        bar.pack(side="bottom", fill="x")
        tk.Frame(bar, bg=COLOR_LINE_DARK, height=1).pack(fill="x")

        body = tk.Frame(bar, bg=COLOR_BAR)
        body.pack(fill="x", padx=20, pady=(12, 14))
        row = tk.Frame(body, bg=COLOR_BAR)
        row.pack(fill="x")
        self.btn_start = PillButton(row, "▶  Старт", command=self._start, variant="accent",
                                     bg=COLOR_BAR, height=46, width=130)
        self.btn_start.pack(side="left")
        self.btn_stop = PillButton(row, "⏹  Стоп", command=self._stop, variant="danger",
                                    bg=COLOR_BAR, height=46, width=130)
        self.btn_stop.configure(state="disabled")
        self.btn_stop.pack(side="left", padx=(10, 0))
        self.progress = RoundedProgressBar(row, bg=COLOR_BAR, height=10)
        self.progress.pack(side="left", fill="x", expand=True, padx=(18, 0))
        self.var_status = tk.StringVar(value="Готово к запуску")
        autowrap(tk.Label(body, textvariable=self.var_status, bg=COLOR_BAR, fg=COLOR_ON_DARK,
                          font=("Segoe UI", 9, "bold"), anchor="w", justify="left")
                 ).pack(fill="x", pady=(8, 0))

    def _field_row(self, parent, pady=(6, 0)):
        row = self._row(parent)
        row.pack(fill="x", pady=pady)
        return row

    def _build_ui(self):
        self.var_provider = tk.StringVar(value=self._active_provider)
        HeaderBar(self).pack(side="top", fill="x")
        self._build_bottom_bar()

        scroller = ScrollableFrame(self)
        scroller.pack(fill="both", expand=True)
        root = scroller.inner

        # ---- Provider: its own block, one container so the choice reads as a choice ----
        self._section(root, "Провайдер ИИ", "Выберите, каким сервисом обрабатывать фото")
        card_provider = Card(root)
        card_provider.pack(fill="x", padx=20)
        SegmentedControl(
            card_provider.body,
            options=[("gemini", PROVIDER_LABELS["gemini"]), ("openai", PROVIDER_LABELS["openai"])],
            variable=self.var_provider, command=self._on_provider_change, bg=COLOR_CARD,
        ).pack(fill="x")

        # ---- Readiness: a row of ringed circles (hidden unless SHOW_READINESS) ----
        self._badges = []
        if SHOW_READINESS:
            self._section(root, "Готовность к запуску", "Отмечается автоматически по мере заполнения")
            steps = tk.Frame(root, bg=COLOR_BG)
            steps.pack(fill="x", padx=20)
            for i, name in enumerate(("Ключ", "Модель", "Фото", "Промпт")):
                steps.columnconfigure(i, weight=1, uniform="steps")
                col = tk.Frame(steps, bg=COLOR_BG)
                col.grid(row=0, column=i, sticky="n")
                badge = StepBadge(col, i + 1)
                badge.pack()
                tk.Label(col, text=name, bg=COLOR_BG, fg=COLOR_ON_DARK, font=FONT_HINT).pack(pady=(6, 0))
                self._badges.append(badge)

        # ---- Connection (API key + model + pause between requests) ----
        self._section(root, "Подключение", "Ключ хранится на этом компьютере только в зашифрованном виде")
        card_conn = Card(root)
        card_conn.pack(fill="x", padx=20)
        b = card_conn.body

        self.var_key_eyebrow = tk.StringVar()
        self._label_var(b, self.var_key_eyebrow).pack(fill="x")
        row_key = self._field_row(b)
        self.var_api_key = tk.StringVar(value=self._keys[self._active_provider])
        self.var_show_key = tk.BooleanVar(value=False)
        self.field_key = InputField(row_key, self.var_api_key, show="*", trailing="eye",
                                    on_trailing=self._toggle_key_visibility)
        self.field_key.pack(side="left", fill="x", expand=True)
        self.entry_api_key = self.field_key.entry
        PillButton(row_key, "Вставить", command=self._paste_api_key,
                   bg=COLOR_CARD).pack(side="left", padx=(10, 0))
        self.var_key_hint = tk.StringVar()
        self._hint_var(b, self.var_key_hint).pack(fill="x", pady=(8, 0))
        if not secure_store.is_available():
            self._hint(b, "⚠ шифрование недоступно (не Windows) — ключ будет храниться в открытом виде"
                       ).pack(fill="x", pady=(6, 0))

        dashed_divider(b).pack(fill="x", pady=16)

        self.var_model_eyebrow = tk.StringVar()
        self._label_var(b, self.var_model_eyebrow).pack(fill="x")
        row_model = self._field_row(b)
        self.var_model = tk.StringVar(value=self._models[self._active_provider])
        self.combo_model = ComboField(row_model, self.var_model,
                                      MODEL_PRESETS_BY_PROVIDER[self._active_provider])
        self.combo_model.pack(side="left", fill="x", expand=True)
        self.btn_refresh_models = PillButton(
            row_model, "Обновить список", command=self._refresh_models, bg=COLOR_CARD,
        )
        self.btn_refresh_models.pack(side="left", padx=(10, 0))
        self.var_model_status = tk.StringVar(value="")
        self.lbl_model_status = autowrap(tk.Label(
            b, textvariable=self.var_model_status, bg=COLOR_CARD, fg=COLOR_MUTED,
            font=FONT_HINT, anchor="w", justify="left",
        ))
        self.lbl_model_status.pack(fill="x", pady=(8, 0))
        self.var_model_hint = tk.StringVar()
        self._hint_var(b, self.var_model_hint).pack(fill="x", pady=(4, 0))
        self._update_provider_labels()

        dashed_divider(b).pack(fill="x", pady=16)

        # Pause between requests (rate-limit protection) - a compact stepper.
        self._label(b, "Задержка в секундах:").pack(fill="x")
        row_delay = self._field_row(b)
        self.var_delay = tk.IntVar(value=int(self.cfg["delay_seconds"]))
        StepperField(row_delay, self.var_delay).pack(side="left")
        self._hint(row_delay, "Пауза между запросами защищает от лимитов; точные лимиты зависят от тарифа."
                   ).pack(side="left", fill="x", expand=True, padx=(12, 0))

        # ---- Folders ----
        self._section(root, "Папки", "Оригиналы не изменяются — результат сохраняется отдельно")
        card_folders = Card(root)
        card_folders.pack(fill="x", padx=20)
        b = card_folders.body
        self._label(b, "Папка с исходными фото (input)").pack(fill="x")
        row_in = self._field_row(b)
        self.var_input = tk.StringVar(value=self.cfg["input_folder"])
        InputField(row_in, self.var_input).pack(side="left", fill="x", expand=True)
        PillButton(row_in, "Обзор...", command=self._pick_input_folder,
                   bg=COLOR_CARD).pack(side="left", padx=(10, 0))

        dashed_divider(b).pack(fill="x", pady=16)

        self._label(b, "Папка для результатов (output)").pack(fill="x")
        row_out = self._field_row(b)
        self.var_output = tk.StringVar(value=self.cfg["output_folder"])
        InputField(row_out, self.var_output).pack(side="left", fill="x", expand=True)
        PillButton(row_out, "Обзор...", command=self._pick_output_folder,
                   bg=COLOR_CARD).pack(side="left", padx=(10, 0))

        # ---- Prompt ----
        self._section(root, "Промпт", "Применяется ко всем фото в папке за один прогон")
        card_prompt = Card(root)
        card_prompt.pack(fill="x", padx=20)
        prompt_box = TextBox(card_prompt.body, lines=5)
        prompt_box.pack(fill="x")
        self.text_prompt = prompt_box.text
        self.text_prompt.insert("1.0", self.cfg["prompt"])

        # ---- Log ----
        self._section(root, "Журнал", "Ход обработки и ошибки")
        card_log = Card(root)
        card_log.pack(fill="x", padx=20)
        log_box = TextBox(card_log.body, lines=10, fill=COLOR_LOG_BG, fg=COLOR_ON_DARK,
                          outline=COLOR_LOG_BG, font=FONT_MONO, scrollbar=True, readonly=True)
        log_box.pack(fill="x")
        self.text_log = log_box.text
        self.text_log.tag_configure("ok", foreground=LOG_OK)
        self.text_log.tag_configure("err", foreground=LOG_ERR)
        self.text_log.tag_configure("warn", foreground=LOG_WARN)
        self.text_log.tag_configure("info", foreground=LOG_INFO)

        # ---- Footer ----
        autowrap(tk.Label(
            root,
            text=(
                "🔒 Приватность: фото и текст промпта отправляются по HTTPS в выбранный вами ИИ-сервис "
                "(Google Gemini или OpenAI). Ключи на этом компьютере хранятся в зашифрованном виде и "
                "никуда, кроме запроса к API выбранного провайдера, не уходят."
            ),
            bg=COLOR_BG, fg=COLOR_ON_DARK_MUTED, font=FONT_HINT, justify="left", anchor="w",
        )).pack(fill="x", padx=20, pady=(24, 26))

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- readiness ----------
    def _steps_tick(self):
        try:
            input_dir = self.var_input.get().strip()
            done = (
                bool(self.var_api_key.get().strip()),
                bool(self.var_model.get().strip()),
                bool(input_dir) and Path(input_dir).is_dir(),
                bool(self.text_prompt.get("1.0", "end").strip()),
            )
            for badge, ok in zip(self._badges, done):
                badge.set_done(ok)
        except (tk.TclError, OSError):
            pass
        self.after(400, self._steps_tick)

    # ---------- clipboard (fixes Ctrl+C/V/X/A under non-Latin keyboard layouts) ----------
    def _fix_clipboard_shortcuts(self):
        self.bind_all("<Control-KeyPress>", self._on_ctrl_keypress, add="+")
        self._attach_context_menu(self.entry_api_key)

    def _on_ctrl_keypress(self, event):
        widget = event.widget
        code = event.keycode
        if code == _VK_V:
            try:
                widget.event_generate("<<Paste>>")
            except tk.TclError:
                pass
            return "break"
        if code == _VK_C:
            try:
                widget.event_generate("<<Copy>>")
            except tk.TclError:
                pass
            return "break"
        if code == _VK_X:
            try:
                widget.event_generate("<<Cut>>")
            except tk.TclError:
                pass
            return "break"
        if code == _VK_A:
            if isinstance(widget, tk.Text):
                widget.tag_add("sel", "1.0", "end-1c")
            elif isinstance(widget, (tk.Entry, ttk.Entry)):
                widget.selection_range(0, "end")
            return "break"
        return None

    def _attach_context_menu(self, entry):
        menu = tk.Menu(self, tearoff=0, bg=COLOR_FIELD, fg=COLOR_TEXT, activebackground=COLOR_ACCENT,
                        activeforeground=COLOR_ON_ACCENT, relief="flat")
        menu.add_command(label="Вырезать", command=lambda: entry.event_generate("<<Cut>>"))
        menu.add_command(label="Копировать", command=lambda: entry.event_generate("<<Copy>>"))
        menu.add_command(label="Вставить", command=lambda: entry.event_generate("<<Paste>>"))
        menu.add_separator()
        menu.add_command(label="Выделить всё", command=lambda: entry.selection_range(0, "end"))

        def show_menu(e):
            menu.tk_popup(e.x_root, e.y_root)

        entry.bind("<Button-3>", show_menu)

    def _paste_api_key(self):
        try:
            clip = self.clipboard_get()
        except tk.TclError:
            messagebox.showinfo("Буфер обмена пуст", "Скопируйте ключ (Ctrl+C) и попробуйте снова.")
            return
        self.var_api_key.set(clip.strip())

    def _on_provider_change(self):
        new = self.var_provider.get()
        prev = self._active_provider
        if prev != new:
            self._keys[prev] = self.var_api_key.get().strip()
            self._models[prev] = self.var_model.get().strip()
        self._active_provider = new
        self.var_api_key.set(self._keys.get(new, ""))
        presets = self._fetched_models.get(new) or MODEL_PRESETS_BY_PROVIDER[new]
        self.combo_model.set_values(presets)
        self.var_model.set(self._models.get(new) or presets[0])
        self.var_model_status.set("")
        self._update_provider_labels()

    def _update_provider_labels(self):
        if self.var_provider.get() == "openai":
            self.var_key_eyebrow.set("API-ключ OpenAI")
            self.var_key_hint.set("Получить ключ: platform.openai.com/api-keys")
            self.var_model_eyebrow.set("Модель OpenAI (GPT Image)")
            self.var_model_hint.set(
                "Список выше — предустановленные варианты. Нажмите «Обновить список», чтобы запросить "
                "у OpenAI актуальные image-модели для вашего ключа. Надёжное редактирование "
                "существующих фото по промпту поддерживают модели семейства gpt-image; DALL·E 3 "
                "редактирование не поддерживает."
            )
        else:
            self.var_key_eyebrow.set("API-ключ Google AI Studio")
            self.var_key_hint.set("Получить ключ: aistudio.google.com/apikey")
            self.var_model_eyebrow.set("Модель Nano Banana")
            self.var_model_hint.set(
                "Список выше — предустановленные варианты. Названия моделей у Google меняются; "
                "нажмите «Обновить список», чтобы запросить у API актуальные модели для вашего ключа."
            )

    def _set_model_status(self, text: str, color: str):
        self.var_model_status.set(text)
        self.lbl_model_status.configure(fg=color)

    def _animate_model_status(self, tick: int):
        if not getattr(self, "_models_loading", False):
            return
        dots = "." * (tick % 4)
        self._set_model_status(f"⏳ Проверяем доступные модели{dots}", COLOR_MUTED)
        self.after(400, lambda: self._animate_model_status(tick + 1))

    def _on_key_changed(self, *_):
        """Whenever a key is typed or pasted, refresh the model list once the
        input settles (debounced, so it doesn't fire on every keystroke)."""
        if self._auto_refresh_job:
            self.after_cancel(self._auto_refresh_job)
            self._auto_refresh_job = None
        if len(self.var_api_key.get().strip()) >= MIN_KEY_LEN_FOR_AUTO_REFRESH:
            self._auto_refresh_job = self.after(AUTO_REFRESH_DEBOUNCE_MS, self._auto_refresh_models)

    def _auto_refresh_models(self):
        self._auto_refresh_job = None
        api_key = self.var_api_key.get().strip()
        provider = self.var_provider.get()
        if len(api_key) < MIN_KEY_LEN_FOR_AUTO_REFRESH or self._auto_tried.get(provider) == api_key:
            return
        if self._models_loading:  # a request is already running; try again right after
            self._auto_refresh_job = self.after(1000, self._auto_refresh_models)
            return
        self._auto_tried[provider] = api_key
        self._refresh_models(auto=True)

    def _refresh_models(self, auto: bool = False):
        api_key = self.var_api_key.get().strip()
        if not api_key:
            self._set_model_status("✗ Сначала укажите API-ключ", COLOR_ERROR)
            return
        provider = self.var_provider.get()
        self.btn_refresh_models.configure(state="disabled")
        self._models_loading = True
        provider_name = "OpenAI" if provider == "openai" else "Google"
        prefix = "Автоматически запрашиваю" if auto else "Запрашиваю"
        self._log(f"{prefix} у {provider_name} список доступных моделей...", "info")
        self._animate_model_status(0)
        worker = self._fetch_openai_models_worker if provider == "openai" else self._fetch_models_worker
        threading.Thread(target=worker, args=(api_key,), daemon=True).start()

    def _fetch_openai_models_worker(self, api_key: str):
        try:
            client = openai.OpenAI(api_key=api_key, timeout=MODEL_LIST_TIMEOUT_S)
            names = []
            for m in client.models.list():
                mid = m.id
                low = mid.lower()
                if "dall-e" in low or ("image" in low and "embed" not in low):
                    names.append(mid)
            names = sorted(set(names))
            self.after(0, lambda: self._on_models_fetched(names, None, "openai"))
        except openai.APITimeoutError:
            err = TimeoutError(
                f"OpenAI не ответил за {MODEL_LIST_TIMEOUT_MS // 1000} сек. Проверьте соединение "
                "(в т.ч. VPN, если он используется) и повторите попытку."
            )
            self.after(0, lambda: self._on_models_fetched(None, err, "openai"))
        except openai.AuthenticationError:
            err = ValueError("Неверный API-ключ OpenAI.")
            self.after(0, lambda: self._on_models_fetched(None, err, "openai"))
        except openai.APIConnectionError as e:
            err = ConnectionError(f"Не удалось соединиться с OpenAI ({type(e).__name__}).")
            self.after(0, lambda: self._on_models_fetched(None, err, "openai"))
        except Exception as e:  # noqa: BLE001
            err = e  # `e` is unbound once the except block ends, before the lambda runs
            self.after(0, lambda: self._on_models_fetched(None, err, "openai"))

    def _fetch_models_worker(self, api_key: str):
        try:
            client = genai.Client(
                api_key=api_key,
                http_options=genai_types.HttpOptions(timeout=MODEL_LIST_TIMEOUT_MS),
            )
            names = []
            for m in client.models.list(config={"page_size": 100}):
                actions = m.supported_actions or []
                name = (m.name or "")
                if name.startswith("models/"):
                    name = name[len("models/"):]
                if "generateContent" in actions and "image" in name.lower():
                    names.append(name)
            missing = [m for m in GEMINI_ALWAYS_LISTED if m not in names]
            names = sorted(set(names) | set(GEMINI_ALWAYS_LISTED))
            if missing:
                self.after(0, lambda: self._log(
                    f"ℹ API не вернул {', '.join(missing)} — модель добавлена в список вручную. "
                    "Если для вашего ключа она недоступна, запрос вернёт ошибку.",
                    "info",
                ))
            self.after(0, lambda: self._on_models_fetched(names, None, "gemini"))
        except httpx.TimeoutException:
            err = TimeoutError(
                f"Google не ответил за {MODEL_LIST_TIMEOUT_MS // 1000} сек. Обычно это значит, что "
                "VPN подключён, но соединение нестабильно/медленное — попробуйте другой сервер VPN "
                "или повторите попытку."
            )
            self.after(0, lambda: self._on_models_fetched(None, err, "gemini"))
        except httpx.NetworkError as e:
            err = ConnectionError(
                f"Не удалось соединиться с Google ({type(e).__name__}). Проверьте, что VPN "
                "действительно подключён и пропускает трафик."
            )
            self.after(0, lambda: self._on_models_fetched(None, err, "gemini"))
        except Exception as e:  # noqa: BLE001
            err = e  # `e` is unbound once the except block ends, before the lambda runs
            self.after(0, lambda: self._on_models_fetched(None, err, "gemini"))

    def _on_models_fetched(self, names, error, provider=None):
        self._models_loading = False
        self.btn_refresh_models.configure(state="normal")

        if provider and provider != self.var_provider.get():
            # The user switched provider while this request was in flight; its
            # answer belongs to the other provider and must not overwrite this list.
            self.var_model_status.set("")
            return

        if error is not None:
            msg = str(error)
            short = msg if len(msg) <= 70 else msg[:67] + "..."
            self._set_model_status(f"✗ Ошибка: {short}", COLOR_ERROR)
            self._log(f"✗ Не удалось получить список моделей: {msg}", "err")
            return

        if not names:
            self._set_model_status("⚠ Моделей не найдено", COLOR_WARNING)
            self._log(
                "⚠ Провайдер не вернул ни одной image-модели для этого ключа. Проверьте, включён ли "
                "доступ к API и есть ли доступ к image-генерации в вашем аккаунте.",
                "warn",
            )
            return

        self._fetched_models[self.var_provider.get()] = list(names)
        self.combo_model.set_values(names)
        if self.var_model.get() not in names:
            self.var_model.set(names[0])
        timestamp = time.strftime("%H:%M:%S")
        self._set_model_status(f"✓ Найдено моделей: {len(names)} · обновлено в {timestamp}", COLOR_SUCCESS)
        self._log(f"✓ Получено {len(names)} доступных image-моделей: {', '.join(names)}", "ok")

    def _toggle_key_visibility(self):
        show = not self.var_show_key.get()
        self.var_show_key.set(show)
        self.entry_api_key.config(show="" if show else "*")
        self.field_key.set_icon_state(show)

    def _pick_input_folder(self):
        folder = filedialog.askdirectory(title="Выберите папку с фото")
        if folder:
            self.var_input.set(folder)
            if not self.var_output.get():
                self.var_output.set(str(Path(folder).parent / (Path(folder).name + "_output")))

    def _pick_output_folder(self):
        folder = filedialog.askdirectory(title="Выберите папку для результатов")
        if folder:
            self.var_output.set(folder)

    # ---------- config persistence ----------
    def _current_cfg(self) -> dict:
        provider = self.var_provider.get()
        # Flush the field currently on screen into the per-provider cache
        # before reading it back out, so switching providers never loses
        # what was typed into the other one.
        self._keys[provider] = self.var_api_key.get().strip()
        self._models[provider] = self.var_model.get().strip()
        gemini_key = self._keys.get("gemini", "")
        openai_key = self._keys.get("openai", "")
        gemini_model = self._models.get("gemini") or GEMINI_MODEL_PRESETS[0]
        openai_model = self._models.get("openai") or OPENAI_MODEL_PRESETS[0]
        active_key = gemini_key if provider == "gemini" else openai_key
        active_model = gemini_model if provider == "gemini" else openai_model
        return {
            "provider": provider,
            # in-memory only; stripped before writing to disk (see save_config)
            "gemini_api_key": gemini_key,
            "openai_api_key": openai_key,
            "api_key": active_key,
            "model": active_model,
            "gemini_api_key_enc": secure_store.encrypt(gemini_key) if secure_store.is_available() and gemini_key else "",
            "openai_api_key_enc": secure_store.encrypt(openai_key) if secure_store.is_available() and openai_key else "",
            "gemini_model": gemini_model,
            "openai_model": openai_model,
            "prompt": self.text_prompt.get("1.0", "end").strip(),
            "input_folder": self.var_input.get().strip(),
            "output_folder": self.var_output.get().strip(),
            "delay_seconds": int(self.var_delay.get()),
        }

    def _on_close(self):
        try:
            save_config(self._current_cfg())
        except Exception:  # noqa: BLE001 - a failed save must never keep the window open
            pass
        self.destroy()

    # ---------- logging ----------
    def _log(self, msg: str, tag: str | None = None):
        self.log_queue.put((msg, tag))

    def _poll_log_queue(self):
        try:
            while True:
                msg, tag = self.log_queue.get_nowait()
                self.text_log.configure(state="normal")
                if tag:
                    self.text_log.insert("end", msg + "\n", tag)
                else:
                    self.text_log.insert("end", msg + "\n")
                self.text_log.see("end")
                self.text_log.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(150, self._poll_log_queue)

    # ---------- run control ----------
    def _start(self):
        cfg = self._current_cfg()
        if not cfg["api_key"]:
            provider_name = "OpenAI" if cfg["provider"] == "openai" else "Google AI Studio"
            messagebox.showerror("Ошибка", f"Укажите API-ключ {provider_name}.")
            return
        if not cfg["model"]:
            messagebox.showerror("Ошибка", "Укажите модель.")
            return
        # Path("") is Path("."), which would silently pass is_dir() and scan the
        # current directory - so an empty field has to be rejected explicitly.
        if not cfg["input_folder"] or not Path(cfg["input_folder"]).is_dir():
            messagebox.showerror("Ошибка", "Укажите существующую папку с исходными фото.")
            return
        input_dir = Path(cfg["input_folder"])
        if not cfg["prompt"]:
            messagebox.showerror("Ошибка", "Введите промпт.")
            return
        if not cfg["output_folder"]:
            cfg["output_folder"] = str(input_dir.parent / (input_dir.name + "_output"))
            self.var_output.set(cfg["output_folder"])

        output_dir = Path(cfg["output_folder"])
        try:
            if output_dir.resolve() == input_dir.resolve():
                messagebox.showerror("Ошибка", "Папка результатов не должна совпадать с папкой input.")
                return
        except OSError:
            pass

        try:
            save_config(cfg)
        except OSError as e:
            self._log(f"⚠ Не удалось сохранить настройки: {e}", "warn")

        try:
            files = sorted(
                p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
        except OSError as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать папку с фото:\n{e}")
            return
        if not files:
            messagebox.showinfo("Нет файлов", "В указанной папке не найдено изображений.")
            return

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            messagebox.showerror("Ошибка", f"Не удалось создать папку для результатов:\n{e}")
            return

        self.stop_event.clear()
        self.progress.configure(maximum=len(files), value=0)
        self.var_status.set(f"0/{len(files)}")
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")

        self.worker = threading.Thread(
            target=self._run_batch,
            args=(cfg, files, output_dir),
            daemon=True,
        )
        self.worker.start()

    def _stop(self):
        self.stop_event.set()
        self._log("Остановка запрошена, дождитесь завершения текущего файла...", "warn")

    def _finish(self, processed: int, total: int, errors: int, abort_note: str = ""):
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        suffix = f" — остановлено: {abort_note}" if abort_note else ""
        self.var_status.set(f"Готово: {processed}/{total} (ошибок: {errors}){suffix}")
        self._log(f"=== Обработка завершена: {processed}/{total}, ошибок: {errors}{suffix} ===", "info")

    # ---------- worker ----------
    @staticmethod
    def _find_duplicate_stems(files: list[Path]) -> set[str]:
        seen, dups = set(), set()
        for p in files:
            key = p.stem.lower()
            (dups if key in seen else seen).add(key)
        return dups

    def _out_stem(self, src_path: Path) -> str:
        """Output file name without extension. Normally the source's own name; when
        a.jpg and a.png sit in the same folder both would become a.png, so the
        source extension is added to keep one result from overwriting the other."""
        if src_path.stem.lower() in self._dup_stems:
            return f"{src_path.stem}_{src_path.suffix.lstrip('.').lower()}"
        return src_path.stem

    def _ui(self, fn, *args, **kwargs):
        """Runs a widget update on the Tk thread (called from the batch thread)."""
        self.after(0, lambda: fn(*args, **kwargs))

    def _run_batch(self, cfg: dict, files: list[Path], output_dir: Path):
        total = len(files)
        processed = 0
        errors = 0
        abort_note = ""
        try:
            provider = cfg["provider"]
            if provider == "openai":
                client = openai.OpenAI(api_key=cfg["api_key"], timeout=GENERATE_TIMEOUT_S)
                process_one = self._process_one_openai
            else:
                client = genai.Client(
                    api_key=cfg["api_key"],
                    http_options=genai_types.HttpOptions(timeout=GENERATE_TIMEOUT_MS),
                )
                process_one = self._process_one_gemini
            prompt = cfg["prompt"]
            model = cfg["model"]
            delay = max(1, cfg["delay_seconds"])
            self._dup_stems = self._find_duplicate_stems(files)

            for idx, path in enumerate(files, start=1):
                if self.stop_event.is_set():
                    self._log("Остановлено пользователем.", "warn")
                    break

                self._log(f"[{idx}/{total}] Обработка {path.name}...")
                try:
                    ok, quota_exhausted = process_one(client, model, prompt, path, output_dir)
                except _AbortBatch as e:
                    processed += 1
                    errors += 1
                    self._log(f"✗ {e} Остановка пакета — остальные файлы завершились бы так же.", "err")
                    abort_note = "неверный ключ или нет доступа к модели"
                    break
                processed += 1
                if not ok:
                    errors += 1

                self._ui(self.progress.configure, value=idx)
                self._ui(self.var_status.set, f"{idx}/{total}")

                if quota_exhausted:
                    if provider == "openai":
                        self._log(
                            "✗ Похоже, закончился баланс/квота OpenAI (insufficient_quota). "
                            "Остановка пакета — повторные попытки не помогут, пополните баланс "
                            "на platform.openai.com/settings/organization/billing.",
                            "err",
                        )
                    else:
                        self._log(
                            "✗ Похоже, исчерпана суточная квота API (RESOURCE_EXHAUSTED, per-day). "
                            "Остановка пакета — повторные попытки прямо сейчас не помогут, "
                            "проверьте лимиты в AI Studio → Usage.",
                            "err",
                        )
                    abort_note = "превышена суточная квота API"
                    break

                if idx < total and not self.stop_event.is_set():
                    # wait() instead of sleep(): "Стоп" interrupts the pause at once
                    self.stop_event.wait(delay)
        except Exception as e:  # noqa: BLE001 - whatever happens, the UI must be released
            self._log(f"✗ Непредвиденная ошибка, пакет остановлен: {e}", "err")
            abort_note = "непредвиденная ошибка"
        finally:
            self.after(0, lambda: self._finish(processed, total, errors, abort_note))

    def _process_one_gemini(self, client, model: str, prompt: str, path: Path, output_dir: Path) -> tuple[bool, bool]:
        """Returns (success, daily_quota_exhausted)."""
        mime = MIME_BY_EXT[path.suffix.lower()]

        try:
            data = path.read_bytes()
        except OSError as e:
            self._log(f"  ✗ Не удалось прочитать файл: {e}", "err")
            return False, False

        image_part = genai_types.Part.from_bytes(data=data, mime_type=mime)

        for attempt in range(1, MAX_RETRIES + 1):
            if self.stop_event.is_set():
                return False, False
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[prompt, image_part],
                )
                return self._save_gemini_image(response, path, output_dir), False
            except genai_errors.ClientError as e:
                if _is_quota_error(e):
                    if _is_daily_quota_error(e):
                        return False, True
                    wait = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * attempt) + random.uniform(0, 2)
                    self._log(
                        f"  ⚠ Лимит запросов (429), повтор через {wait:.0f} сек "
                        f"({attempt}/{MAX_RETRIES})...",
                        "warn",
                    )
                    self.stop_event.wait(wait)
                    continue
                if _is_fatal_gemini_error(e):
                    raise _AbortBatch(f"Google отклонил запрос: {e}")
                self._log(f"  ✗ Ошибка API: {e}", "err")
                return False, False
            except genai_errors.ServerError as e:
                wait = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * attempt) + random.uniform(0, 2)
                self._log(f"  ⚠ Ошибка сервера ({e.code}), повтор через {wait:.0f} сек...", "warn")
                self.stop_event.wait(wait)
                continue
            except httpx.TimeoutException:
                wait = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * attempt) + random.uniform(0, 2)
                self._log(
                    f"  ⚠ Google не ответил за {GENERATE_TIMEOUT_MS // 1000} сек (похоже на "
                    f"нестабильный VPN), повтор через {wait:.0f} сек ({attempt}/{MAX_RETRIES})...",
                    "warn",
                )
                self.stop_event.wait(wait)
                continue
            except httpx.NetworkError as e:
                wait = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * attempt) + random.uniform(0, 2)
                self._log(
                    f"  ⚠ Сетевая ошибка ({type(e).__name__}, возможно VPN оборвал соединение), "
                    f"повтор через {wait:.0f} сек...",
                    "warn",
                )
                self.stop_event.wait(wait)
                continue
            except Exception as e:  # noqa: BLE001
                self._log(f"  ✗ Ошибка: {e}", "err")
                return False, False

        self._log(f"  ✗ Превышено число попыток для {path.name}", "err")
        return False, False

    def _process_one_openai(self, client, model: str, prompt: str, path: Path, output_dir: Path) -> tuple[bool, bool]:
        """Returns (success, quota_exhausted)."""
        for attempt in range(1, MAX_RETRIES + 1):
            if self.stop_event.is_set():
                return False, False
            try:
                with open(path, "rb") as f:
                    response = client.images.edit(model=model, image=f, prompt=prompt)
                return self._save_openai_image(response, path, output_dir), False
            except openai.RateLimitError as e:
                if _is_openai_quota_exhausted(e):
                    return False, True
                wait = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * attempt) + random.uniform(0, 2)
                self._log(
                    f"  ⚠ Лимит запросов (429), повтор через {wait:.0f} сек "
                    f"({attempt}/{MAX_RETRIES})...",
                    "warn",
                )
                self.stop_event.wait(wait)
                continue
            except (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError) as e:
                raise _AbortBatch(f"OpenAI отклонил запрос ({e.status_code}): {e.message}")
            except openai.APIStatusError as e:
                if 500 <= (e.status_code or 0) < 600:
                    wait = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * attempt) + random.uniform(0, 2)
                    self._log(f"  ⚠ Ошибка сервера OpenAI ({e.status_code}), повтор через {wait:.0f} сек...", "warn")
                    self.stop_event.wait(wait)
                    continue
                self._log(f"  ✗ Ошибка API OpenAI: {e}", "err")
                return False, False
            except openai.APITimeoutError:
                wait = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * attempt) + random.uniform(0, 2)
                self._log(
                    f"  ⚠ OpenAI не ответил за {GENERATE_TIMEOUT_MS // 1000} сек, "
                    f"повтор через {wait:.0f} сек ({attempt}/{MAX_RETRIES})...",
                    "warn",
                )
                self.stop_event.wait(wait)
                continue
            except openai.APIConnectionError as e:
                wait = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * attempt) + random.uniform(0, 2)
                self._log(
                    f"  ⚠ Сетевая ошибка ({type(e).__name__}), повтор через {wait:.0f} сек...",
                    "warn",
                )
                self.stop_event.wait(wait)
                continue
            except Exception as e:  # noqa: BLE001
                self._log(f"  ✗ Ошибка: {e}", "err")
                return False, False

        self._log(f"  ✗ Превышено число попыток для {path.name}", "err")
        return False, False

    def _save_openai_image(self, response, src_path: Path, output_dir: Path) -> bool:
        try:
            items = response.data or []
            if not items:
                self._log("  ✗ Модель не вернула изображение (возможно, промпт был отклонён).", "err")
                return False
            item = items[0]
            out_path = output_dir / f"{self._out_stem(src_path)}.png"
            if getattr(item, "b64_json", None):
                out_path.write_bytes(base64.b64decode(item.b64_json))
                self._log(f"  ✓ Сохранено: {out_path.name}", "ok")
                return True
            if getattr(item, "url", None):
                raw = httpx.get(item.url, timeout=60).content
                out_path.write_bytes(raw)
                self._log(f"  ✓ Сохранено: {out_path.name}", "ok")
                return True
            self._log("  ✗ Ответ OpenAI не содержит изображения.", "err")
            return False
        except Exception as e:  # noqa: BLE001
            self._log(f"  ✗ Не удалось сохранить результат: {e}", "err")
            return False

    def _save_gemini_image(self, response, src_path: Path, output_dir: Path) -> bool:
        try:
            candidates = response.candidates or []
            for cand in candidates:
                if not cand.content or not cand.content.parts:
                    continue
                for part in cand.content.parts:
                    inline = getattr(part, "inline_data", None)
                    if inline and inline.data:
                        ext = ".png" if "png" in (inline.mime_type or "") else ".jpg"
                        out_path = output_dir / f"{self._out_stem(src_path)}{ext}"
                        out_path.write_bytes(inline.data)
                        self._log(f"  ✓ Сохранено: {out_path.name}", "ok")
                        return True
            self._log("  ✗ Модель не вернула изображение (возможно, промпт был отклонён).", "err")
            return False
        except Exception as e:  # noqa: BLE001
            self._log(f"  ✗ Не удалось сохранить результат: {e}", "err")
            return False


if __name__ == "__main__":
    App().mainloop()
