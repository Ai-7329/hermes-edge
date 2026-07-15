# win-ui-mcp — Windows native UI automation as a single-purpose MCP tool

A thin **UIAutomation-based** MCP server that gives a text-only Hermes agent
hands and eyes on **native Windows apps** — without a vision model. It exposes
the Windows accessibility tree as structured text (element name / control type
/ AutomationId), so the model acts on elements *by reference*, not by pixel
coordinates.

**Why this shape (and not Everfern whole, or pixel + VLM):**
- No LLM runs in this server — it only enumerates and actuates. Your single
  local `llama-server` stays the only inference loop, so hermes-edge's
  single-slot stability (R1) is untouched. This server never talks to the model.
- UIAutomation is structured, so a text model (e.g. `Qwen3.6-35b`) can operate
  it. Pixel-coordinate computer-use would require a vision model you don't have.
- Single purpose: only "operate a Windows UI." No Electron, no VM, no second
  agent runtime competing for the GPU.

## Install (on the Windows box)

```powershell
pip install uiautomation "mcp[cli]" pillow
```

Python 3.10+. `pillow` is only needed for the optional `screenshot()` fallback.

## Smoke test (no Hermes)

```powershell
python C:\path\to\win_ui_mcp.py --selftest
```

Prints the top-level windows and the foreground window's element tree. If your
target app shows almost no elements, it is UIA-opaque (some Electron / custom-
drawn UIs) — see Limitations.

## Wire into Hermes

Add to the **active** config (`%LOCALAPPDATA%\hermes\config.yaml`, or
`%HERMES_HOME%\config.yaml`):

```yaml
mcp_servers:
  winui:
    command: python
    args: ["C:\\path\\to\\win_ui_mcp.py"]
```

Restart Hermes. The tools load as `winui.*` and are deferred behind
`tool_search` (they do not bloat the fixed header — good for the edge profile).

Keep approvals manual so the model can't drive the desktop unattended:

```yaml
approvals:
  mode: manual
```

## Tools

| Tool | Purpose |
|---|---|
| `list_windows()` | enumerate top-level windows (start here) |
| `dump_tree(window, max_depth, max_nodes)` | numbered element tree of a window — the "screenshot" the model reads |
| `click(window, ref\|name\|automation_id, button, double)` | click / Invoke a control |
| `set_text(window, text, ref\|name\|automation_id)` | set an edit field (Value pattern, atomic) |
| `get_text(window, ref\|name\|automation_id)` | read a control's value/name |
| `send_keys(keys)` | global keystrokes: `{Ctrl}s`, `{Enter}`, `{Alt}{F4}` |
| `focus_window(window)` | bring a window to the foreground |
| `screenshot(path)` | best-effort PNG (fallback for UIA-opaque apps) |

Typical loop the model runs: `list_windows` → `dump_tree` → act by `ref` →
`dump_tree` again to confirm.

## Limitations (honest)

- **UIA-opaque apps.** Some Electron/Chromium and custom-drawn apps expose a
  thin or empty tree. For those, structured automation can't see the controls;
  you'd need pixel+vision (out of scope here) or the app's own accessibility
  mode enabled.
- **Stale refs.** A `ref` is an index into the last `dump_tree` of that window.
  If the UI changed since, an action errors and asks you to re-dump — it will
  not click the wrong element silently. `name`/`automation_id` queries are
  more robust across small changes.
- **COM threading.** UIAutomation is COM; if you hit apartment/threading errors
  under load, run the server single-threaded (it is one desktop, one actuator —
  serial operation is correct anyway).
- **It controls the whole desktop.** Run under manual approvals; treat it like
  giving the agent your mouse and keyboard.
