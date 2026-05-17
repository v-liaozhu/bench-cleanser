"""Tests for bench_cleanser.parsing.patch_parser."""

from bench_cleanser.parsing.patch_parser import get_files_from_patch, parse_patch

# ── parse_patch ───────────────────────────────────────────────────────


def test_empty_patch():
    assert parse_patch("") == []
    assert parse_patch(None) == []
    assert parse_patch("   \n  ") == []


def test_single_hunk():
    diff = """\
diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,4 @@ def greet
 hello
-world
+universe
+!
"""
    hunks = parse_patch(diff)
    assert len(hunks) == 1
    h = hunks[0]
    assert h.file_path == "foo.py"
    assert h.hunk_index == 0
    assert h.added_lines == ["universe", "!"]
    assert h.removed_lines == ["world"]
    assert h.context_lines == ["hello"]
    assert h.function_context == "def greet"


def test_multi_hunk_same_file():
    diff = """\
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,2 +1,2 @@
-old1
+new1
@@ -10,2 +10,2 @@
-old2
+new2
"""
    hunks = parse_patch(diff)
    assert len(hunks) == 2
    assert all(h.file_path == "a.py" for h in hunks)
    assert hunks[0].hunk_index == 0
    assert hunks[1].hunk_index == 1


def test_multi_file_diff():
    diff = """\
diff --git a/x.py b/x.py
--- a/x.py
+++ b/x.py
@@ -1,1 +1,1 @@
-a
+b
diff --git a/y.py b/y.py
--- a/y.py
+++ b/y.py
@@ -1,1 +1,1 @@
-c
+d
"""
    hunks = parse_patch(diff)
    assert len(hunks) == 2
    assert hunks[0].file_path == "x.py"
    assert hunks[1].file_path == "y.py"


def test_binary_and_no_newline_markers():
    diff = """\
diff --git a/img.png b/img.png
Binary files a/img.png and b/img.png differ
diff --git a/f.py b/f.py
--- a/f.py
+++ b/f.py
@@ -1,1 +1,1 @@
-old
+new
\\ No newline at end of file
"""
    hunks = parse_patch(diff)
    assert len(hunks) == 1
    assert hunks[0].file_path == "f.py"
    assert hunks[0].added_lines == ["new"]


# ── get_files_from_patch ──────────────────────────────────────────────


def test_get_files_empty():
    assert get_files_from_patch("") == []
    assert get_files_from_patch(None) == []


def test_get_files_multi():
    diff = """\
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1 +1 @@
-x
+y
diff --git a/b.py b/b.py
--- a/b.py
+++ b/b.py
@@ -1 +1 @@
-x
+y
"""
    files = get_files_from_patch(diff)
    assert files == ["a.py", "b.py"]


def test_get_files_dedup():
    diff = """\
diff --git a/same.py b/same.py
@@ -1 +1 @@
-a
+b
diff --git a/same.py b/same.py
@@ -10 +10 @@
-c
+d
"""
    files = get_files_from_patch(diff)
    assert files == ["same.py"]
