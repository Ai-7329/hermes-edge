#!/usr/bin/env python3
"""win-ui-mcp — a thin Windows UI-automation MCP server (UIAutomation-based).

Single purpose: give a text-only agent hands and eyes on native Windows apps
WITHOUT a vision model. Instead of screenshots + pixel coordinates, it exposes
the Windows UIAutomation accessibility tree as structured text — the model
reads element names / control types / AutomationIds and acts on them by
reference. No LLM runs inside this server; it only enumerates and actuates,
so the agent's single local model stays the only inference loop (keeps
hermes-edge's single-slot stability intact — this server never touches the
llama-server).

Tools:
  list_windows()                         -> top-level windows
  dump_tree(window, max_depth, max_nodes)-> numbered element tree of a window
  click(window, ref|name|automation_id, button, double) -> click/invoke
  set_text(window, text, ref|name|automation_id)        -> set an edit value
  get_text(window, ref|name|automation_id)              -> read a value/name
  send_keys(keys)                        -> global keystrokes ("{Ctrl}s", "{Enter}")
  focus_window(window)                   -> bring a window to the foreground
  screenshot(path)                       -> best-effort PNG (fallback for UIA-opaque apps)

Requires (Windows, Python 3.10+):
  pip install uiautomation "mcp[cli]"
  # screenshot() also wants: pip install pillow

Run standalone smoke test (no MCP):
  python win_ui_mcp.py --selftest

Wire into hermes-edge (~/.hermes config.yaml, or %LOCALAPPDATA%\\hermes\\config.yaml):
  mcp_servers:
    winui:
      command: python
      args: ["C:\\\\path\\\\to\\\\win_ui_mcp.py"]

Then restart Hermes. Keep approvals manual (approvals.mode: manual) — this
server can drive the whole desktop.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

# uiautomation is Windows-only (wraps the UIAutomation COM API via comtypes).
try:
    import uiautomation as auto
except Exception as exc:  # pragma: no cover - import guard
    sys.stderr.write(
        "win-ui-mcp: could not import 'uiautomation'. On Windows run:\n"
        "  pip install uiautomation \"mcp[cli]\"\n"
        f"import error: {exc}\n"
    )
    raise

# Per-window cache of the last dumped element list, so click/set_text can
# reference an element by its integer index from the most recent dump_tree.
# Refs go stale if the UI changes between dump and act — actions then raise a
# clear "re-dump" error rather than clicking the wrong thing.
_TREE_CACHE: Dict[str, List["auto.Control"]] = {}

_DEFAULT_TIMEOUT = 3  # seconds uiautomation waits for a control to appear


# ── helpers ─────────────────────────────────────────────────────────────────

def _wkey(window: str) -> str:
    return (window or "").strip().lower()


def _find_window(window: str) -> "auto.Control":
    """Return the first top-level window whose title contains ``window``
    (case-insensitive). Raises ValueError if none matches."""
    target = _wkey(window)
    root = auto.GetRootControl()
    for w in root.GetChildren():
        try:
            name = (w.Name or "")
        except Exception:
            name = ""
        if target and target in name.lower():
            return w
    # Also allow matching by exact automation search (some windows are lazy)
    raise ValueError(
        f"no top-level window matches {window!r}. Call list_windows() first."
    )


def _patterns(ctrl: "auto.Control") -> List[str]:
    """Names of the interaction patterns a control supports (best-effort)."""
    names = []
    for pat_name, getter in (
        ("Invoke", "GetInvokePattern"),
        ("Value", "GetValuePattern"),
        ("Toggle", "GetTogglePattern"),
        ("ExpandCollapse", "GetExpandCollapsePattern"),
        ("SelectionItem", "GetSelectionItemPattern"),
        ("LegacyIAccessible", "GetLegacyIAccessiblePattern"),
    ):
        try:
            if getattr(ctrl, getter)() is not None:
                names.append(pat_name)
        except Exception:
            pass
    return names


def _describe(ctrl: "auto.Control", ref: int) -> Dict[str, Any]:
    try:
        r = ctrl.BoundingRectangle
        rect = [r.left, r.top, r.right, r.bottom]
    except Exception:
        rect = None
    val = ""
    try:
        vp = ctrl.GetValuePattern()
        if vp is not None:
            val = vp.Value or ""
    except Exception:
        pass
    try:
        name = ctrl.Name or ""
    except Exception:
        name = ""
    try:
        aid = ctrl.AutomationId or ""
    except Exception:
        aid = ""
    try:
        cls = ctrl.ClassName or ""
    except Exception:
        cls = ""
    try:
        enabled = bool(ctrl.IsEnabled)
    except Exception:
        enabled = True
    return {
        "ref": ref,
        "control": ctrl.ControlTypeName,
        "name": name,
        "automation_id": aid,
        "class": cls,
        "value": val[:120],
        "enabled": enabled,
        "rect": rect,
        "patterns": _patterns(ctrl),
    }


def _walk(root: "auto.Control", max_depth: int, max_nodes: int) -> List["auto.Control"]:
    """Depth-first collect of descendant controls, bounded."""
    out: List["auto.Control"] = []
    stack = [(root, 0)]
    # skip the window node itself (index 0 is the window); include its subtree
    first = True
    while stack and len(out) < max_nodes:
        node, depth = stack.pop()
        if not first:
            out.append(node)
        first = False
        if depth >= max_depth:
            continue
        try:
            children = node.GetChildren()
        except Exception:
            children = []
        # reverse so DFS visits in natural order
        for c in reversed(children):
            stack.append((c, depth + 1))
    return out


def _resolve(
    window: str,
    ref: Optional[int],
    name: Optional[str],
    automation_id: Optional[str],
    control_type: Optional[str],
) -> "auto.Control":
    if ref is not None:
        lst = _TREE_CACHE.get(_wkey(window))
        if not lst:
            raise ValueError(
                f"no cached tree for {window!r}. Call dump_tree() before using ref."
            )
        if ref < 0 or ref >= len(lst):
            raise ValueError(f"ref {ref} out of range (0..{len(lst)-1}); re-dump_tree.")
        return lst[ref]
    # query resolution: re-dump and match the first element
    lst = _TREE_CACHE.get(_wkey(window)) or []
    if not lst:
        win = _find_window(window)
        lst = _walk(win, max_depth=12, max_nodes=1000)
        _TREE_CACHE[_wkey(window)] = lst
    for c in lst:
        try:
            if automation_id and (c.AutomationId or "") != automation_id:
                continue
            if name and name.lower() not in (c.Name or "").lower():
                continue
            if control_type and control_type.lower() not in c.ControlTypeName.lower():
                continue
            if name or automation_id or control_type:
                return c
        except Exception:
            continue
    raise ValueError(
        "no element matched "
        f"(name={name!r}, automation_id={automation_id!r}, control_type={control_type!r}). "
        "Call dump_tree() and act by ref."
    )


# ── MCP server ───────────────────────────────────────────────────────────────

def build_server():
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("win-ui")

    @mcp.tool()
    def list_windows() -> str:
        """List visible top-level windows (title, class, pid, rect). Start here."""
        root = auto.GetRootControl()
        out = []
        for w in root.GetChildren():
            try:
                name = w.Name or ""
            except Exception:
                name = ""
            if not name:
                continue
            try:
                r = w.BoundingRectangle
                rect = [r.left, r.top, r.right, r.bottom]
            except Exception:
                rect = None
            out.append({
                "title": name,
                "control": w.ControlTypeName,
                "class": getattr(w, "ClassName", "") or "",
                "pid": getattr(w, "ProcessId", 0) or 0,
                "rect": rect,
            })
        return json.dumps(out, ensure_ascii=False, indent=2)

    @mcp.tool()
    def dump_tree(window: str, max_depth: int = 8, max_nodes: int = 200) -> str:
        """Dump a window's UIAutomation element tree as a numbered list.

        `window` is a case-insensitive substring of the window title. Each
        element has a `ref` (int) usable in click()/set_text()/get_text(),
        plus name, automation_id, control type, current value, and the
        interaction patterns it supports. This replaces a screenshot: read
        the elements and act by ref.
        """
        win = _find_window(window)
        try:
            win.SetActive()
        except Exception:
            pass
        controls = _walk(win, max_depth=max_depth, max_nodes=max_nodes)
        _TREE_CACHE[_wkey(window)] = controls
        described = [_describe(c, i) for i, c in enumerate(controls)]
        return json.dumps(
            {"window": win.Name, "count": len(described), "elements": described},
            ensure_ascii=False, indent=2,
        )

    @mcp.tool()
    def click(
        window: str,
        ref: Optional[int] = None,
        name: Optional[str] = None,
        automation_id: Optional[str] = None,
        control_type: Optional[str] = None,
        button: str = "left",
        double: bool = False,
    ) -> str:
        """Click an element. Identify it by `ref` (from dump_tree) or by
        name/automation_id/control_type. Buttons/menu items are activated via
        the Invoke pattern when available (more reliable than a raw click);
        otherwise a real mouse click at the element's clickable point."""
        ctrl = _resolve(window, ref, name, automation_id, control_type)
        # Prefer Invoke for a plain left single-click on invokable controls.
        if button == "left" and not double:
            try:
                inv = ctrl.GetInvokePattern()
                if inv is not None:
                    ctrl.SetActive() if hasattr(ctrl, "SetActive") else None
                    inv.Invoke()
                    return json.dumps({"ok": True, "action": "invoke", "element": _describe(ctrl, ref if ref is not None else -1)})
            except Exception:
                pass
        try:
            if button == "right":
                ctrl.RightClick(waitTime=0)
            elif double:
                ctrl.DoubleClick(waitTime=0)
            else:
                ctrl.Click(waitTime=0)
            return json.dumps({"ok": True, "action": f"{'double_' if double else ''}{button}_click"})
        except Exception as exc:
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}. Re-run dump_tree (ref may be stale)."})

    @mcp.tool()
    def set_text(
        window: str,
        text: str,
        ref: Optional[int] = None,
        name: Optional[str] = None,
        automation_id: Optional[str] = None,
    ) -> str:
        """Set an edit control's text. Uses the Value pattern (atomic, reliable)
        when supported; otherwise focuses the element and types the string."""
        ctrl = _resolve(window, ref, name, automation_id, None)
        try:
            vp = ctrl.GetValuePattern()
            if vp is not None:
                vp.SetValue(text)
                return json.dumps({"ok": True, "action": "set_value"})
        except Exception:
            pass
        try:
            ctrl.SetFocus()
            # {Ctrl}a clears existing content, then type the new text literally.
            ctrl.SendKeys("{Ctrl}a", waitTime=0)
            ctrl.SendKeys(text, waitTime=0)
            return json.dumps({"ok": True, "action": "sendkeys"})
        except Exception as exc:
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @mcp.tool()
    def get_text(
        window: str,
        ref: Optional[int] = None,
        name: Optional[str] = None,
        automation_id: Optional[str] = None,
    ) -> str:
        """Read an element's value (Value pattern) or its Name."""
        ctrl = _resolve(window, ref, name, automation_id, None)
        val = ""
        try:
            vp = ctrl.GetValuePattern()
            if vp is not None:
                val = vp.Value or ""
        except Exception:
            pass
        if not val:
            try:
                val = ctrl.Name or ""
            except Exception:
                val = ""
        return json.dumps({"ok": True, "text": val})

    @mcp.tool()
    def send_keys(keys: str) -> str:
        """Send keystrokes to the foreground window. Supports specials, e.g.
        "{Ctrl}s", "{Enter}", "{Alt}{F4}", "Hello{Space}world". See the
        uiautomation SendKeys syntax."""
        try:
            auto.SendKeys(keys, waitTime=0)
            return json.dumps({"ok": True})
        except Exception as exc:
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @mcp.tool()
    def focus_window(window: str) -> str:
        """Bring a window to the foreground (activate + restore)."""
        win = _find_window(window)
        try:
            win.SetActive()
            win.SetTopmost(True)
            win.SetTopmost(False)
            return json.dumps({"ok": True, "window": win.Name})
        except Exception as exc:
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @mcp.tool()
    def screenshot(path: str = "") -> str:
        """Best-effort PNG capture of the whole screen (fallback for UIA-opaque
        apps like some Electron/custom-drawn UIs). Needs Pillow. Returns the
        saved path — a text model can't read the image, but it helps debugging
        or a future vision endpoint."""
        try:
            import tempfile, os
            out = path or os.path.join(tempfile.gettempdir(), "win_ui_shot.png")
            auto.GetRootControl().CaptureToImage(out)
            return json.dumps({"ok": True, "path": out})
        except Exception as exc:
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}. pip install pillow"})

    return mcp


def _selftest() -> int:
    """Enumerate windows and dump the foreground window's tree — no MCP."""
    root = auto.GetRootControl()
    print("Top-level windows:")
    for w in root.GetChildren():
        try:
            if w.Name:
                print(f"  - {w.ControlTypeName}: {w.Name!r}")
        except Exception:
            pass
    fg = auto.GetForegroundControl()
    print(f"\nForeground: {fg.Name!r} ({fg.ControlTypeName})")
    controls = _walk(fg.GetTopLevelControl(), max_depth=6, max_nodes=40)
    for i, c in enumerate(controls):
        d = _describe(c, i)
        print(f"  [{i}] {d['control']} name={d['name']!r} aid={d['automation_id']!r} patterns={d['patterns']}")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    build_server().run()
