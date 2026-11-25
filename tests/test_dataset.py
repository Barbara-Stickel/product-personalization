from pathlib import Path

from PIL import Image

from product_lora.dataset import validate_dataset


def test_dataset_validation_detects_duplicate_and_missing_identifier(tmp_path: Path) -> None:
    image = Image.new("RGB", (600, 600), "green")
    image.save(tmp_path / "one.png")
    image.save(tmp_path / "two.png")
    (tmp_path / "one.txt").write_text("zkttent in a field")
    (tmp_path / "two.txt").write_text("a tent in a field")
    report = validate_dataset(tmp_path, "zkttent")
    assert not report.valid
    assert any("Duplicate" in error for error in report.errors)
    assert any("omits identifier" in error for error in report.errors)
