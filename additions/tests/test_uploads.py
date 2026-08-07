"""Focused tests for IPSAE upload storage and safe zip extraction."""

from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import uploads


class FakeUpload:
    def __init__(self, items: list[dict] | dict):
        self.value = items


def _v8_upload(name: str, content: bytes) -> FakeUpload:
    return FakeUpload([{"name": name, "content": content, "size": len(content)}])


class UploadTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.files_dir = self.root / "upload" / "files"
        self.folders_dir = self.root / "upload" / "folders"
        self.files_dir.mkdir(parents=True)
        self.folders_dir.mkdir(parents=True)

        self.patches = [
            mock.patch.object(uploads, "UPLOAD_DIR", self.root / "upload"),
            mock.patch.object(uploads, "UPLOAD_FILES_DIR", self.files_dir),
            mock.patch.object(uploads, "UPLOAD_FOLDERS_DIR", self.folders_dir),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in self.patches:
            patcher.stop()
        self._tmpdir.cleanup()

    def test_save_structure_pdb_and_cif(self) -> None:
        for name, content in (("model.pdb", b"ATOM\n"), ("model.cif", b"data_\n")):
            path = uploads.save_single_upload(_v8_upload(name, content), "structure")
            self.assertEqual(path, (self.files_dir / name).resolve())
            self.assertEqual(path.read_bytes(), content)

    def test_save_pae_json_and_npz(self) -> None:
        for name, content in (("pae.json", b"{}"), ("pae.npz", b"NPZ")):
            path = uploads.save_single_upload(_v8_upload(name, content), "pae")
            self.assertEqual(path, (self.files_dir / name).resolve())
            self.assertEqual(path.read_bytes(), content)

    def test_reject_unsupported_extension(self) -> None:
        with self.assertRaises(uploads.UploadError):
            uploads.save_single_upload(_v8_upload("notes.txt", b"nope"), "structure")

    def test_overwrite_same_basename(self) -> None:
        first = uploads.save_single_upload(_v8_upload("model.cif", b"v1"), "structure")
        second = uploads.save_single_upload(_v8_upload("model.cif", b"v2"), "structure")
        self.assertEqual(first, second)
        self.assertEqual(second.read_bytes(), b"v2")

    def test_extract_zip_to_folders_stem(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("fold_x/fold_x_model_0.cif", "cif")
            zf.writestr("fold_x/fold_x_full_data_0.json", "{}")
        dest = uploads.extract_uploaded_zip(_v8_upload("AF3_outputs.zip", buf.getvalue()))
        self.assertEqual(dest, (self.folders_dir / "AF3_outputs").resolve())
        self.assertTrue((dest / "fold_x" / "fold_x_model_0.cif").is_file())

    def test_unwrap_matching_top_level_dir(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("AF3_outputs/fold_x/fold_x_model_0.cif", "cif")
            zf.writestr("AF3_outputs/fold_x/fold_x_full_data_0.json", "{}")
        dest = uploads.extract_uploaded_zip(_v8_upload("AF3_outputs.zip", buf.getvalue()))
        self.assertTrue((dest / "fold_x" / "fold_x_model_0.cif").is_file())
        self.assertFalse((dest / "AF3_outputs").exists())

    def test_reupload_deletes_existing_folder(self) -> None:
        dest = self.folders_dir / "AF3_outputs"
        dest.mkdir()
        stale = dest / "stale.txt"
        stale.write_text("old")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("fold_x/model.cif", "cif")
        out = uploads.extract_uploaded_zip(_v8_upload("AF3_outputs.zip", buf.getvalue()))
        self.assertEqual(out, dest.resolve())
        self.assertFalse(stale.exists())
        self.assertTrue((out / "fold_x" / "model.cif").is_file())

    def test_reject_zip_slip(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../bad.txt", "evil")
        zip_path = self.files_dir / "slip.zip"
        zip_path.write_bytes(buf.getvalue())
        dest = self.folders_dir / "slip"
        dest.mkdir()
        with self.assertRaises(uploads.UploadError):
            uploads.safe_extract_zip(zip_path, dest)

    def test_reject_absolute_archive_path(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            info = zipfile.ZipInfo("/tmp/evil.txt")
            zf.writestr(info, "evil")
        zip_path = self.files_dir / "abs.zip"
        zip_path.write_bytes(buf.getvalue())
        dest = self.folders_dir / "abs"
        dest.mkdir()
        with self.assertRaises(uploads.UploadError):
            uploads.safe_extract_zip(zip_path, dest)

    def test_reject_symlink_entry(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            info = zipfile.ZipInfo("link.txt")
            info.create_system = 3
            info.external_attr = (0o120777 & 0xFFFF) << 16
            zf.writestr(info, "target")
        zip_path = self.files_dir / "link.zip"
        zip_path.write_bytes(buf.getvalue())
        dest = self.folders_dir / "link"
        dest.mkdir()
        with self.assertRaises(uploads.UploadError):
            uploads.safe_extract_zip(zip_path, dest)

    def test_list_uploaded_folders_only_under_upload_folders(self) -> None:
        (self.folders_dir / "keep_me").mkdir()
        (self.folders_dir / "also").mkdir()
        (self.files_dir / "not_a_folder_listing").write_text("x")
        listed = uploads.list_uploaded_folders()
        self.assertEqual(listed, ["also", "keep_me"])

    def test_uploaded_folder_path(self) -> None:
        path = uploads.uploaded_folder_path("AF3_outputs")
        self.assertEqual(path, (self.folders_dir / "AF3_outputs").resolve())

    def test_v7_upload_dict_format(self) -> None:
        widget = FakeUpload(
            {
                "model.cif": {
                    "content": b"data_",
                    "metadata": {"name": "model.cif"},
                }
            }
        )
        path = uploads.save_single_upload(widget, "structure")
        self.assertEqual(path.read_bytes(), b"data_")


class FolderPickerPathTests(unittest.TestCase):
    def test_folder_path_value(self) -> None:
        from folder_picker import NO_FOLDERS, folder_path_value

        class FakeDropdown:
            def __init__(self, value):
                self.value = value

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "alpha").mkdir()
            path = folder_path_value(FakeDropdown("alpha"), base=base)
            self.assertEqual(path, (base / "alpha").resolve())
            self.assertIsNone(folder_path_value(FakeDropdown(NO_FOLDERS), base=base))


class PathsRootTests(unittest.TestCase):
    def test_resolve_project_root_points_at_freebindcraft(self) -> None:
        from paths import ROOT, UPLOAD_FILES_DIR, UPLOAD_FOLDERS_DIR, resolve_project_root

        root = resolve_project_root()
        self.assertEqual(root, ROOT)
        self.assertTrue((root / "bindcraft.py").exists() or (root / "ipsae.py").exists())
        self.assertEqual(UPLOAD_FILES_DIR, ROOT / "upload" / "files")
        self.assertEqual(UPLOAD_FOLDERS_DIR, ROOT / "upload" / "folders")


if __name__ == "__main__":
    unittest.main()
