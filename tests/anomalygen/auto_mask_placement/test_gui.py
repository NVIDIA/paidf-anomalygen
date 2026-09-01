# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AMP GUI backend's request-facing boundaries.

The GUI is developer tooling that runs on loopback, but every one of its routes is unauthenticated
and several of them touch the filesystem on behalf of the caller. That makes two properties worth
pinning rather than assuming:

* it stays same-origin — no CORS headers, so a page the user happens to have open cannot drive the
  API on their behalf;
* a caller cannot choose where the pipeline writes — ``/api/output-directory/set`` takes a path
  from the request body and creates it, so the boundary is the whole protection.

The heavier ROI/model paths need real checkpoints and a GPU and are not exercised here.
"""

import io
import logging
import os
import pathlib
import re

import pytest
from PIL import Image

from anomalygen.auto_mask_placement.gui.backend import app as app_module
from anomalygen.auto_mask_placement.gui.backend.app import _resolve_output_directory, app


@pytest.fixture
def client():
    """Authenticated client: the launch token arrives as the cookie a real browser would hold."""
    c = app.test_client()
    c.set_cookie(app_module._TOKEN_COOKIE, app_module._AUTH_TOKEN)
    return c


@pytest.fixture
def anonymous_client():
    """A local process that never opened the launcher URL."""
    return app.test_client()


@pytest.fixture
def confined(tmp_path, monkeypatch):
    """Restrict the allowed output roots to a temp dir so escapes are unambiguous."""
    monkeypatch.setattr(
        "anomalygen.auto_mask_placement.gui.backend.app._ALLOWED_OUTPUT_ROOTS",
        (str(tmp_path),),
    )
    return tmp_path


# --- same-origin posture --------------------------------------------------------------------------


def test_same_origin_request_succeeds(client):
    assert client.get("/api/health").status_code == 200


def test_no_cors_header_is_returned_to_a_cross_origin_request(client):
    """`CORS(app)` allowed every origin; with no authentication that let any page drive the API."""
    response = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert response.headers.get("Access-Control-Allow-Origin") is None


def test_the_app_registers_no_cors_extension():
    assert "cors" not in {name.lower() for name in app.extensions}


# --- output-directory confinement -----------------------------------------------------------------


def test_accepts_a_path_inside_the_allowed_root(confined):
    wanted = confined / "results" / "run1"
    assert _resolve_output_directory(str(wanted)) == str(wanted)


def test_accepts_the_root_itself(confined):
    assert _resolve_output_directory(str(confined)) == str(confined)


@pytest.mark.parametrize(
    "escape",
    [
        "/etc/cron.d",
        "~/.ssh",
        "{root}/../escape",
        "{root}/results/../../escape",
    ],
)
def test_rejects_paths_outside_the_allowed_root(confined, escape):
    with pytest.raises(ValueError, match="must be inside"):
        _resolve_output_directory(escape.format(root=str(confined)))


def test_rejects_a_symlink_escaping_the_allowed_root(tmp_path, monkeypatch):
    """A prefix check alone would pass this: the link sits inside the root but points outside it."""
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    link = root / "sneaky"
    link.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(
        "anomalygen.auto_mask_placement.gui.backend.app._ALLOWED_OUTPUT_ROOTS",
        (str(root),),
    )
    with pytest.raises(ValueError, match="must be inside"):
        _resolve_output_directory(str(link))


def test_resolution_does_not_create_anything(confined):
    """Rejection must leave no trace, and validation must not be what makes the directory."""
    target = confined / "not_created_yet"
    assert _resolve_output_directory(str(target)) == str(target)
    assert not target.exists()


# --- the endpoint, not just the resolver ----------------------------------------------------------


def test_endpoint_rejects_an_escaping_path_without_creating_it(client, confined):
    escape = confined.parent / "escape_via_endpoint"
    response = client.post("/api/output-directory/set", json={"directory": str(escape)})

    assert response.status_code == 400
    assert "must be inside" in response.get_json()["error"]
    assert not escape.exists(), "a rejected path must not be created"


def test_endpoint_accepts_a_path_inside_the_allowed_root(client, confined):
    wanted = confined / "results" / "endpoint_run"
    response = client.post("/api/output-directory/set", json={"directory": str(wanted)})

    assert response.status_code == 200
    assert response.get_json()["output_directory"] == str(wanted)
    assert wanted.is_dir(), "an accepted path is created for the caller"


def test_endpoint_requires_a_directory(client):
    response = client.post("/api/output-directory/set", json={"directory": "  "})
    assert response.status_code == 400


# --- session isolation ----------------------------------------------------------------------------
# Session state is keyed by the X-AMP-Session header so two browsers do not collide on fixed
# filenames. That keying is also what stops one caller reading another's uploads.


def test_health_reports_status_without_requiring_a_session(client):
    body = client.get("/api/health").get_json()
    assert body["status"] == "ok"
    assert set(body) >= {"status", "roi_models_ready", "roi_init_in_progress"}


def test_sessions_are_keyed_by_header(client, confined):
    """Two session ids must not share an output override."""
    wanted = confined / "session_a"
    client.post("/api/output-directory/set", json={"directory": str(wanted)}, headers={"X-AMP-Session": "a"})

    a = client.get("/api/output-directory/get", headers={"X-AMP-Session": "a"}).get_json()
    b = client.get("/api/output-directory/get", headers={"X-AMP-Session": "b"}).get_json()

    assert a["output_directory"] == str(wanted)
    assert a["is_temp"] is False
    assert b["output_directory"] != str(wanted), "a second session must not inherit the override"
    assert b["is_temp"] is True


def test_output_directory_reset_returns_to_a_temp_directory(client, confined):
    wanted = confined / "run"
    client.post("/api/output-directory/set", json={"directory": str(wanted)}, headers={"X-AMP-Session": "reset"})
    client.post("/api/output-directory/reset", headers={"X-AMP-Session": "reset"})

    body = client.get("/api/output-directory/get", headers={"X-AMP-Session": "reset"}).get_json()
    assert body["is_temp"] is True
    assert body["output_directory"] != str(wanted)


# --- upload validation ----------------------------------------------------------------------------


def test_upload_rejects_a_request_with_no_file(client):
    response = client.post("/api/upload/input-image", data={})
    assert response.status_code == 400
    assert "No file provided" in response.get_json()["error"]


def test_upload_rejects_an_empty_filename(client):
    response = client.post("/api/upload/input-image", data={"file": (io.BytesIO(b""), "")})
    assert response.status_code == 400


@pytest.mark.parametrize("name", ["payload.py", "shell.sh", "index.html", "archive.tar.gz"])
def test_upload_rejects_a_disallowed_extension(client, name):
    """Only image/json types are accepted; a planted script must not land in the upload directory."""
    response = client.post("/api/upload/input-image", data={"file": (io.BytesIO(b"x"), name)})
    assert response.status_code != 200


def test_upload_accepts_a_png_and_reports_its_size(client):
    buffer = io.BytesIO()
    Image.new("RGB", (12, 7), (10, 20, 30)).save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post(
        "/api/upload/input-image",
        data={"file": (buffer, "sample.png")},
        headers={"X-AMP-Session": "upload"},
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["width"] == 12 and body["height"] == 7


def test_upload_strips_directory_traversal_from_the_filename(client):
    """werkzeug's secure_filename must flatten the name so an upload cannot escape its directory."""
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (1, 2, 3)).save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post(
        "/api/upload/input-image",
        data={"file": (buffer, "../../escape.png")},
        headers={"X-AMP-Session": "traversal"},
    )
    assert response.status_code == 200

    stored = app_module._sessions["traversal"]["input_image"]
    assert ".." not in stored
    assert os.path.realpath(stored).startswith(os.path.realpath(app_module._UPLOAD_BASE))


# --- static + misc --------------------------------------------------------------------------------


def test_index_is_served(client):
    assert client.get("/").status_code == 200


def test_browse_submask_folder_requires_files(client):
    response = client.post("/api/browse/submask-folder", data={})
    assert response.status_code == 400


def test_roi_default_config_is_returned(client):
    response = client.get("/api/roi/config/default")
    assert response.status_code == 200
    assert isinstance(response.get_json(), dict)


# --- filename and pool-path boundaries -------------------------------------------------------------


def test_submask_selection_flattens_a_traversing_filename(client, tmp_path, monkeypatch):
    """The browse endpoint stores file.filename verbatim, so the write side must sanitise it."""
    import base64
    import io as _io

    from PIL import Image as _Image

    buffer = _io.BytesIO()
    _Image.new("RGB", (4, 4), (9, 9, 9)).save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode()

    monkeypatch.setattr(
        app_module,
        "_sessions",
        {
            "sel": {
                "_sid": "sel",
                "submask_candidates": [{"filename": "../../../escape.png", "image": payload, "width": 4, "height": 4}],
            }
        },
    )
    response = client.post("/api/select/submask", json={"index": 0}, headers={"X-AMP-Session": "sel"})

    assert response.status_code == 200
    upload_root = os.path.realpath(app_module._UPLOAD_BASE)
    written = [
        os.path.join(dirpath, name)
        for dirpath, _dirs, names in os.walk(upload_root)
        for name in names
        if "escape" in name
    ]
    assert written, "the upload should still be written, just flattened"
    for path in written:
        assert os.path.realpath(path).startswith(upload_root), "upload escaped the upload folder"


@pytest.mark.parametrize("outside", ["/etc", "/root"])
def test_submask_pool_rejects_a_path_outside_the_allowed_root(client, outside):
    """Same primitive as output-directory: a path from the request that gets listed and read."""
    response = client.post("/api/validate/submask-pool", json={"path": outside})
    assert response.status_code == 400
    assert "must be inside" in response.get_json()["error"]


@pytest.mark.parametrize("outside", ["/etc", "/root"])
def test_overlay_heatmap_rejects_a_path_outside_the_allowed_root(client, outside):
    response = client.post("/api/overlay-heatmap", json={"directory": outside})
    assert response.status_code == 400
    assert "must be inside" in response.get_json()["error"]


def test_overlay_heatmap_does_not_return_masks_from_outside_the_allowed_root(client, confined, tmp_path):
    """The response embeds the composited masks as base64, so an unconfined walk exfiltrates them.

    Distinct from the parametrized case above, which only pins the status code: /etc and /root hold
    no auto_placed_mask*.png, so unconfined they still fail — just as a 500 "none found" after
    walking the tree, with no data returned. Here the file exists and is readable, so unconfined the
    endpoint answers 200 with the mask encoded in the body. That is the property being pinned.
    """
    elsewhere = tmp_path.parent / "outside_the_root"
    elsewhere.mkdir()
    Image.new("L", (8, 8), 255).save(elsewhere / "auto_placed_mask_secret.png")

    response = client.post("/api/overlay-heatmap", json={"directory": str(elsewhere)})

    assert response.status_code == 400
    body = response.get_json()
    assert "must be inside" in body["error"]
    assert "image" not in body


def test_overlay_heatmap_reads_masks_inside_the_allowed_root(client, confined):
    """The boundary must not break the legitimate call it is wrapped around."""
    Image.new("L", (8, 8), 255).save(confined / "auto_placed_mask_0.png")

    response = client.post("/api/overlay-heatmap", json={"directory": str(confined)})

    assert response.status_code == 200
    assert response.get_json()["num_masks"] == 1


# --- session-id path confinement --------------------------------------------------------------------


def test_session_id_is_reduced_to_one_path_segment():
    """The id becomes a directory name, so a traversing value must not survive into a path."""
    assert app_module._session_key("abc-123") == "abc-123"
    for hostile in ("../../etc", "a/b", "..", ".", "x\x00y"):
        key = app_module._session_key(hostile)
        assert "/" not in key and key not in (".", "..")


def test_session_id_traversal_cannot_escape_the_upload_root(client):
    """The session id was a fourth request-controlled path input, and the only unchecked one."""
    upload_root = os.path.realpath(app_module._UPLOAD_BASE)
    escape = os.path.realpath(os.path.join(upload_root, "..", "amp_session_escape_probe"))
    assert not os.path.exists(escape), "stale probe from an earlier run"

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (1, 2, 3)).save(buffer, format="PNG")
    buffer.seek(0)
    client.post(
        "/api/upload/input-image",
        data={"file": (buffer, "probe.png")},
        headers={"X-AMP-Session": "../amp_session_escape_probe"},
        content_type="multipart/form-data",
    )

    assert not os.path.exists(escape), "session id escaped the upload root"


def test_distinct_session_ids_still_get_distinct_directories():
    """Hashing the unsafe ones must not collapse separate sessions onto one directory."""
    assert app_module._session_key("../a") != app_module._session_key("../b")


# --- launch-token access control ------------------------------------------------------------------
# Loopback is not an access control: any local process can reach these routes. The token gates
# /api/*, and delivering it as a SameSite=Strict cookie is also what stops a foreign page forging
# requests the operator's browser would otherwise authenticate.


def test_api_requires_the_launch_token(anonymous_client):
    response = anonymous_client.post("/api/output-directory/reset")
    assert response.status_code == 401
    assert "token" in response.get_json()["error"]


def test_health_stays_open_for_readiness_polling(anonymous_client):
    """The page polls this before it has the cookie, and it returns booleans only."""
    assert anonymous_client.get("/api/health").status_code == 200


def test_static_ui_stays_open(anonymous_client):
    assert anonymous_client.get("/").status_code == 200


def test_token_may_also_be_supplied_as_a_header(anonymous_client):
    response = anonymous_client.get("/api/output-directory/get", headers={"X-AMP-Token": app_module._AUTH_TOKEN})
    assert response.status_code == 200


def test_a_wrong_token_is_refused(anonymous_client):
    response = anonymous_client.get("/api/output-directory/get", headers={"X-AMP-Token": "not-the-token"})
    assert response.status_code == 401


def test_index_exchanges_a_valid_token_for_a_samesite_cookie(anonymous_client):
    """SameSite=Strict is the CSRF control: the browser withholds it on cross-site requests."""
    response = anonymous_client.get(f"/?token={app_module._AUTH_TOKEN}")
    cookie = response.headers.get("Set-Cookie", "")

    assert app_module._TOKEN_COOKIE in cookie
    assert "SameSite=Strict" in cookie
    assert "HttpOnly" in cookie


def test_index_ignores_a_wrong_token(anonymous_client):
    response = anonymous_client.get("/?token=wrong")
    assert app_module._TOKEN_COOKIE not in response.headers.get("Set-Cookie", "")


@pytest.mark.parametrize(("base_url", "secure"), [("http://localhost", False), ("https://localhost", True)])
def test_cookie_carries_the_secure_flag_only_over_https(anonymous_client, base_url, secure):
    """Hardcoding Secure would drop the cookie on the plain-HTTP bind the launcher uses."""
    response = anonymous_client.get(f"/?token={app_module._AUTH_TOKEN}", base_url=base_url)

    assert ("Secure" in response.headers.get("Set-Cookie", "")) is secure


# --- credential redaction -------------------------------------------------------------------------


def test_error_responses_have_token_shaped_values_redacted(client, monkeypatch):
    """Handlers return str(e), and an upstream client error can carry a token in its text."""

    def _raise(*_args, **_kwargs):
        raise RuntimeError("401 from https://hf.co: bad credential hf_abcdefghij0123456789ABCDEFGHIJ")

    monkeypatch.setattr(app_module, "output_folder", _raise)
    response = client.get("/api/download/results-info")

    body = response.get_data(as_text=True)
    assert "hf_abcdefghij0123456789ABCDEFGHIJ" not in body
    assert "[redacted]" in body


@pytest.mark.parametrize(("status", "survives"), [(200, True), (500, False)])
def test_only_error_responses_are_rewritten(status, survives):
    """Redaction is scoped to errors. The 500 case is the control: same body, so it proves the
    pattern would have matched had the guard not stopped it."""
    token_shaped = "hf_" + "A" * 24  # matches _SECRET_PATTERN; kept low-entropy for secret scans
    response = app_module.app.response_class(
        f'{{"note": "{token_shaped}"}}', status=status, mimetype="application/json"
    )

    body = app_module._redact_secrets_from_errors(response).get_data(as_text=True)

    assert (token_shaped in body) is survives


def test_non_ascii_token_is_refused_not_crashed(anonymous_client):
    """Headers decode as latin-1, and compare_digest raises TypeError on a non-ASCII str.

    Without the .isascii() guard Flask turns that into a 500. The request was denied either
    way, but a 500 misreports a clean rejection as a server fault.
    """
    for supplied in ("töken", "ÿ" * 8):
        assert anonymous_client.get("/api/output-directory/get", headers={"X-AMP-Token": supplied}).status_code == 401
        assert (
            anonymous_client.get("/api/output-directory/get", headers={"Cookie": f"amp_token={supplied}"}).status_code
            == 401
        )


def test_access_log_redacts_the_token_from_the_url():
    """Werkzeug logs the request line, so the /?token=... exchange would persist the token."""
    record = logging.LogRecord(
        "werkzeug",
        logging.INFO,
        __file__,
        0,
        '%s - - [%s] "%s"',
        ("127.0.0.1", "now", f"GET /?token={app_module._AUTH_TOKEN} HTTP/1.1"),
        None,
    )
    app_module._RedactTokenInAccessLog().filter(record)

    rendered = record.getMessage()
    assert app_module._AUTH_TOKEN not in rendered
    assert "token=[redacted]" in rendered


@pytest.mark.parametrize("route", ["/api/output-directory/get", "/"])
def test_no_route_500s_on_a_non_ascii_token(anonymous_client, route):
    """Every input the token arrives on goes through one guarded comparison.

    The gate was guarded first and the /?token= exchange was not, so the same TypeError
    still reached the root URL. Both now route through _token_matches.
    """
    for supplied in ("é", "%C3%A9", "ÿ" * 8):
        assert anonymous_client.get(f"{route}?token={supplied}", headers={"X-AMP-Token": supplied}).status_code != 500


def test_non_ascii_query_token_declines_the_cookie_without_erroring(anonymous_client):
    response = anonymous_client.get("/?token=%C3%A9")
    assert response.status_code == 200
    assert app_module._TOKEN_COOKIE not in response.headers.get("Set-Cookie", "")


# --- the token must not outlive its one-time exchange -------------------------------------------
# Access logs are scrubbed, but the URL itself persists in browser history and would ride along as
# a Referer. The exchange redirects to a clean URL so the cookie, not the URL, carries the credential.


def test_valid_token_exchange_redirects_to_a_url_without_the_token(anonymous_client):
    response = anonymous_client.get(f"/?token={app_module._AUTH_TOKEN}")

    assert response.status_code == 303
    location = response.headers["Location"]
    assert "token" not in location, "the redirect target must not carry the token"
    assert app_module._AUTH_TOKEN not in location
    # The exchange is still what mints the cookie — the redirect must not cost us the login.
    assert app_module._TOKEN_COOKIE in response.headers.get("Set-Cookie", "")


def test_redirected_client_lands_authenticated(anonymous_client):
    """Following the redirect must leave the browser able to call the API."""
    anonymous_client.get(f"/?token={app_module._AUTH_TOKEN}", follow_redirects=True)

    assert anonymous_client.get("/api/output-directory/get").status_code == 200


def test_index_sends_no_referrer(anonymous_client):
    """A no-referrer page cannot leak the one URL that did carry the token."""
    for response in (
        anonymous_client.get("/"),
        anonymous_client.get(f"/?token={app_module._AUTH_TOKEN}"),
    ):
        assert response.headers.get("Referrer-Policy") == "no-referrer"


def test_generate_failure_returns_no_traceback(client, monkeypatch):
    """A 500 must carry the message alone: a traceback names server paths and internals.

    Every other handler here already returns ``{"error": str(e)}``; this pins that /api/generate
    does the same. The body below is missing the augmentation keys, so building the config raises.
    """
    monkeypatch.setattr(app_module, "session_data", {"submask": object(), "roi_json": object()})

    response = client.post("/api/generate", json={})

    assert response.status_code == 500
    assert "traceback" not in response.get_json()


def test_every_remote_script_is_pinned_and_integrity_checked():
    """The page runs same-origin with an API that writes to disk, so a CDN swap must not execute.

    Both halves matter: ``integrity`` is what verifies the bytes, and a floating version makes the
    hash unmaintainable — so the URL has to name an exact release for the hash to mean anything.
    """
    index = pathlib.Path(app_module.__file__).parent.parent / "frontend" / "public" / "index.html"
    tags = re.findall(r'<script\b[^>]*\bsrc="https?://[^"]+"[^>]*>', index.read_text())
    assert tags, "expected the page to load libraries from a CDN"

    for tag in tags:
        assert 'integrity="sha384-' in tag, f"no subresource integrity on: {tag}"
        assert "crossorigin" in tag, f"integrity is ignored without crossorigin: {tag}"
        assert re.search(r"@\d+\.\d+\.\d+", tag), f"version not pinned: {tag}"
